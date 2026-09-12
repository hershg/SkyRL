"""Transformer layer with Manifold-Constrained Hyper-Connections (mHC), MoE sub-layers included.

megatron-core's own ``HyperConnectionTransformerLayer`` rejects MoE MLP submodules (mHC + MoE is
only reachable there by wrapping MoE as a ``HybridStack`` layer). This one supports dense *and*
MoE MLPs, which GLM-5.3-Flash needs (mHC on every block, MoE in all but the first layer).

The n residual streams travel between layers as ``[s, b, n * hidden_size]``;
``TransformerBlock`` owns the block-boundary expand (replication) and contract (unweighted mean)
whenever ``enable_mhc_connections`` is set, so this layer only implements the per-sub-layer
residual update.

Parameter names (``self_attention_hyper_connection.*`` / ``mlp_hyper_connection.*`` with
``mapping_proj.weight``, ``bias``, ``alpha_pre``, ``alpha_post``, ``alpha_res``) match upstream so
checkpoints and HF bridges carry over unchanged.
"""

from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from skyrl.backends.skyrl_train.workers.megatron.mcore_ext.hyper_connection import (
    RMSNormInputHyperConnectionModule,
)


class HyperConnectionTransformerLayer(TransformerLayer):
    """``TransformerLayer`` whose residual paths are mHC n-stream hyper-connections.

    For each sub-layer F (self-attention, MLP/MoE) with n-stream input x ([s, b, n*C]):

        h_pre, h_post, h_res = mapping(x)           # per-token, fp32, Sinkhorn-projected h_res
        y = F(norm(h_pre @ x))                      # single-stream sub-layer
        x = h_res^T @ x + h_post ⊗ y                # residual mix + broadcast of the update

    Cross-attention is not supported.
    """

    supports_mhc_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            **kwargs,
        )
        if not getattr(config, "enable_mhc_connections", False):
            raise ValueError(
                "HyperConnectionTransformerLayer requires enable_mhc_connections=True."
            )
        if not isinstance(self.cross_attention, IdentityOp):
            raise ValueError(
                "HyperConnectionTransformerLayer does not support cross-attention."
            )
        if (
            self.recompute_input_layernorm
            or self.recompute_pre_mlp_layernorm
            or self.recompute_mlp
        ):
            raise NotImplementedError(
                "Selective 'layernorm'/'mlp' activation recompute is not supported by "
                "HyperConnectionTransformerLayer."
            )
        if config.fp32_residual_connection:
            raise NotImplementedError(
                "fp32_residual_connection is not supported with mHC streams."
            )

        # megatron-core builds these from ``submodules.self_attention_hyper_connection`` /
        # ``mlp_hyper_connection`` specs; ``TransformerLayerSubmodules`` has no such slots, so
        # build them directly. Their parameters are fp32-marked, so the cast below only applies
        # to anything a future variant may add in the activation dtype.
        hyper_connection_cls = (
            RMSNormInputHyperConnectionModule
            if getattr(config, "mhc_norm_eps_inside_sqrt", False)
            else HyperConnectionModule
        )
        with torch.device(torch.cuda.current_device()):
            self.self_attention_hyper_connection = hyper_connection_cls(
                config=config, layer_number=self.layer_number
            )
            self.mlp_hyper_connection = hyper_connection_cls(
                config=config, layer_number=self.layer_number
            )
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(
                self.self_attention_hyper_connection, config.params_dtype
            )
            convert_module_to_dtype_except_fp32_marked(
                self.mlp_hyper_connection, config.params_dtype
            )

        self.mhc_checkpoint_input_layernorm = not isinstance(
            self.input_layernorm, IdentityOp
        )
        self.mhc_checkpoint_pre_mlp_layernorm = not isinstance(
            self.pre_mlp_layernorm, IdentityOp
        )
        self._mhc_recompute_manager = None

    def __call__(self, *args, **kwargs):
        # CheckpointWithoutOutputManager is not a supported CUDA-graph kwarg. Match MCore's mHC
        # layer by removing it before the inherited graph dispatch and reading it from the layer.
        self._mhc_recompute_manager = kwargs.pop("mhc_recompute_manager", None)
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _reject_residual_returning_norm(layernorm_output, norm_name: str):
        """mHC needs the n-stream residual captured before aggregation, so a layernorm that
        also returns a (single-stream) residual cannot be used."""
        if isinstance(layernorm_output, tuple):
            raise ValueError(
                f"HyperConnectionTransformerLayer does not support a {norm_name} that returns an "
                "(output, residual) tuple."
            )
        return layernorm_output

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        mhc_recompute_manager=None,
        **kwargs,
    ):
        """Run one mHC block over n-stream hidden states (``[s, b, n * hidden_size]``)."""
        if context is not None:
            raise ValueError(
                "HyperConnectionTransformerLayer does not support cross-attention context."
            )
        mhc_recompute_manager = self._mhc_recompute_manager or mhc_recompute_manager

        # Self-attention site.
        residual = hidden_states
        aggregated, h_res, h_post = self.self_attention_hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        if mhc_recompute_manager is not None and self.mhc_checkpoint_input_layernorm:
            input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            input_layernorm_output = input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, aggregated
            )
        else:
            input_layernorm_checkpoint = None
            input_layernorm_output = self.input_layernorm(aggregated)
        input_layernorm_output = self._reject_residual_returning_norm(
            input_layernorm_output, "input_layernorm"
        )
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        if input_layernorm_checkpoint is not None:
            input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                attention_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_recompute_manager,
            )

        # MLP / MoE site.
        residual = hidden_states
        aggregated, h_res, h_post = self.mlp_hyper_connection(
            hidden_states, mhc_recompute_manager=mhc_recompute_manager
        )
        if mhc_recompute_manager is not None and self.mhc_checkpoint_pre_mlp_layernorm:
            pre_mlp_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                ckpt_manager=mhc_recompute_manager
            )
            pre_mlp_layernorm_output = pre_mlp_layernorm_checkpoint.checkpoint(
                self.pre_mlp_layernorm, aggregated
            )
        else:
            pre_mlp_layernorm_checkpoint = None
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(aggregated)
        pre_mlp_layernorm_output = self._reject_residual_returning_norm(
            pre_mlp_layernorm_output, "pre_mlp_layernorm"
        )
        pre_mlp_layernorm_output, moe_padding_mask, moe_unflatten_mbs = (
            self._maybe_unflatten_for_moe(
                pre_mlp_layernorm_output, padding_mask, packed_seq_params
            )
        )
        mlp_output_with_bias = self._run_mlp(
            pre_mlp_layernorm_output,
            residual,
            moe_padding_mask,
            inference_context,
        )
        if moe_unflatten_mbs is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = self._maybe_reflatten_from_moe(
                mlp_output, packed_seq_params, moe_unflatten_mbs
            )
            mlp_output_with_bias = (mlp_output, mlp_bias)
        is_last_in_recompute_block = bool(
            mhc_recompute_manager is not None
            and getattr(
                mhc_recompute_manager, "is_last_layer_in_recompute_block", False
            )
        )
        mhc_mlp_bda_manager = (
            None if is_last_in_recompute_block else mhc_recompute_manager
        )
        if pre_mlp_layernorm_checkpoint is not None and mhc_mlp_bda_manager is not None:
            pre_mlp_layernorm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_hyper_connection.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                mlp_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
                mhc_mlp_bda_manager,
            )

        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )
        return output, None
