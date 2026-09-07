"""GLM 5.3 LoRA parity on a multi-node B300 Ray cluster.

Run with:
SKYRL_GLM53_SHARED_DIR=/shared/skyrl-tests \
uv run --isolated --extra dev --extra megatron -- \
pytest -s -m b300 tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_glm53_lora_logprobs.py

For a two-GPU Qwen dry run, also set:
SKYRL_GLM53_MODEL=Qwen/Qwen2.5-1.5B-Instruct

For a prestarted multi-node Ray cluster, also set:
SKYRL_GLM53_RAY_ADDRESS=auto
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SamplingParams, SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    init_worker_with_type,
)

MODEL = "zai-org/GLM-5.3-BF16"
SMALL_DRY_RUN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POLICY_GPUS = 8
INFERENCE_TP = 8
MAX_GENERATE_LENGTH = 128
MEGATRON_MEAN_DIFF_THRESHOLD = 5e-2
VLLM_MEAN_DIFF_THRESHOLD = 3e-1
LORA_NOISE_SEED = 42
LORA_NOISE_STD = 1e-2
FINAL_DENSE_LORA_NOISE_STD = 3.0
MAX_FULL_HASH_NUMEL = 1_000_000
MIN_UPDATED_LOGPROB_DIFF = 1e-5
MIN_DIRECT_UPDATE_MEAN = 0.1
MIN_DIRECT_UPDATE_COSINE = 0.8
MIN_DIRECT_UPDATE_SCALE = 0.5
MAX_DIRECT_UPDATE_SCALE = 1.5
MAX_DIRECT_UPDATE_RELATIVE_MEAN_ERROR = 0.5

TEST_PROMPTS = [
    "What is 2 + 3? Answer with only the number.",
    "Name the largest planet in our solar system. Answer briefly.",
    "Complete the sequence: 1, 1, 2, 3, 5, __.",
    "Write one short sentence explaining why ice floats on water.",
]

MEGATRON_LORA_TARGET_MODULES = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]

VLLM_LORA_TARGET_MODULES = [
    "fused_qkv_a_proj",
    "q_a_proj",
    "q_b_proj",
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "experts",
]


def _get_sample_indices(
    numel: int, sample_count: int, device: torch.device
) -> torch.Tensor:
    if sample_count == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if sample_count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    positions = torch.arange(sample_count, dtype=torch.long, device=device)
    return torch.div(positions * (numel - 1), sample_count - 1, rounding_mode="floor")


def _get_sampled_tensor_receipt(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().reshape(-1)
    sample_count = min(64, flat.numel())
    if sample_count:
        indices = _get_sample_indices(flat.numel(), sample_count, flat.device)
        sample = flat.index_select(0, indices).float().cpu()
    else:
        sample = torch.empty(0, dtype=torch.float32)
    receipt = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "sample_sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
        "sample_mean": sample.mean().item() if sample_count else 0.0,
        "sample_l2": torch.linalg.vector_norm(sample).item() if sample_count else 0.0,
        "sample_values": sample[:8].tolist(),
    }
    if tensor.numel() <= MAX_FULL_HASH_NUMEL:
        tensor_bytes = tensor.detach().contiguous().view(torch.uint8).cpu()
        receipt["full_sha256"] = hashlib.sha256(
            tensor_bytes.numpy().tobytes()
        ).hexdigest()
    return receipt


def test_sample_indices_stay_in_bounds_above_float32_integer_range():
    numel = 15_380_406_246
    indices = _get_sample_indices(numel, 64, torch.device("cpu"))

    assert indices[0].item() == 0
    assert indices[-1].item() == numel - 1
    assert torch.all(indices[1:] > indices[:-1])


def _get_weight_receipt(weight) -> list[dict] | dict | None:
    if weight is None:
        return None
    if isinstance(weight, (list, tuple)):
        return [_get_sampled_tensor_receipt(tensor) for tensor in weight]
    return _get_sampled_tensor_receipt(weight)


def _get_expert_weight_receipt(weight) -> list[dict] | dict | None:
    def add_expert_slices(tensor: torch.Tensor) -> dict:
        receipt = _get_sampled_tensor_receipt(tensor)
        if tensor.ndim >= 3:
            expert_indices = sorted({0, tensor.shape[0] // 2, tensor.shape[0] - 1})
            receipt["representative_experts"] = {
                str(index): _get_sampled_tensor_receipt(tensor[index])
                for index in expert_indices
            }
        return receipt

    if weight is None:
        return None
    if isinstance(weight, (list, tuple)):
        return [add_expert_slices(tensor) for tensor in weight]
    return add_expert_slices(weight)


def _get_final_layer_names(names: list[str], marker: str) -> list[str]:
    matched = []
    for name in names:
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        if marker in name and layer_match is not None:
            matched.append((int(layer_match.group(1)), name))
    if not matched:
        return []
    final_layer = max(layer for layer, _ in matched)
    return sorted(name for layer, name in matched if layer == final_layer)


def _get_representative_expert_names(names: list[str]) -> list[str]:
    final_layer_names = _get_final_layer_names(names, ".mlp.experts")
    names_by_expert: dict[int, list[str]] = {}
    for name in final_layer_names:
        expert_match = re.search(r"\.experts\.(\d+)\.", name)
        if expert_match is not None:
            names_by_expert.setdefault(int(expert_match.group(1)), []).append(name)
    if names_by_expert:
        expert_ids = sorted(names_by_expert)
        representative_ids = {
            expert_ids[0],
            expert_ids[len(expert_ids) // 2],
            expert_ids[-1],
        }
        return sorted(
            name
            for expert_id in representative_ids
            for name in names_by_expert[expert_id]
        )
    if len(final_layer_names) <= 12:
        return final_layer_names
    indices = torch.linspace(0, len(final_layer_names) - 1, steps=12).long()
    return [final_layer_names[index] for index in indices]


class _InspectableInferenceWorkerWrap(NewInferenceWorkerWrap):
    def inspect_glm53_lora_buffers(self) -> dict:
        update_scope = os.environ.get("SKYRL_GLM53_UPDATE_SCOPE", "all")
        final_dense = update_scope == "final_dense"
        module_marker = ".self_attn.o_proj" if final_dense else ".mlp.experts"
        worker_manager = self.model_runner.lora_manager
        adapter_manager = worker_manager._adapter_manager
        adapter_ids = sorted(adapter_manager.list_adapters())
        receipt = {
            "hostname": socket.gethostname(),
            "update_scope": update_scope,
            "adapter_ids": adapter_ids,
            "slot_layout": adapter_manager.lora_index_to_id,
            "enable_mixed_moe_lora_format": adapter_manager._enable_mixed_moe_lora_format,
            "enable_moe_shared_loras": adapter_manager._enable_moe_shared_loras,
            "is_3d_moe_model": adapter_manager._is_3d_moe_model,
            "use_ep": adapter_manager._use_ep,
        }
        if len(adapter_ids) != 1:
            return receipt

        adapter_id = adapter_ids[0]
        slot = adapter_manager.lora_index_to_id.index(adapter_id)
        adapter = adapter_manager.get_adapter(adapter_id)
        assert adapter is not None
        cached_names = _get_final_layer_names(list(adapter.loras), module_marker)
        receipt["cached_adapter"] = {
            name: {
                "lora_a": _get_weight_receipt(adapter.loras[name].lora_a),
                "lora_b": (
                    _get_weight_receipt(adapter.loras[name].lora_b)
                    if final_dense
                    else _get_expert_weight_receipt(adapter.loras[name].lora_b)
                ),
                "scaling": adapter.loras[name].scaling,
            }
            for name in cached_names
        }

        module_names = _get_final_layer_names(
            list(adapter_manager.modules), module_marker
        )
        receipt["kernel_buffers"] = {}
        for name in module_names:
            module = adapter_manager.modules[name]
            if final_dense:
                receipt["kernel_buffers"][name] = {
                    "module_type": type(module).__name__,
                    "tp_rank": module.tp_rank,
                    "tp_size": module.tp_size,
                    "lora_a": _get_weight_receipt(module.lora_a_stacked[0][slot, 0]),
                    "lora_b": _get_weight_receipt(module.lora_b_stacked[0][slot, 0]),
                }
                continue
            expert_map = module.base_layer.routed_experts.expert_map
            receipt["kernel_buffers"][name] = {
                "module_type": type(module).__name__,
                "ep_rank": module.ep_rank,
                "global_num_experts": module.global_num_experts,
                "local_num_experts": module.local_num_experts,
                "tp_rank": module.tp_rank,
                "tp_size": module.tp_size,
                "enable_moe_shared_loras": module.enable_moe_shared_loras,
                "expert_map": (
                    expert_map.detach().cpu().tolist()
                    if expert_map is not None
                    else None
                ),
                "w13_lora_a": [
                    _get_expert_weight_receipt(tensor[slot])
                    for tensor in module.w13_lora_a_stacked
                ],
                "w13_lora_b": [
                    _get_expert_weight_receipt(tensor[slot])
                    for tensor in module.w13_lora_b_stacked
                ],
                "w2_lora_a": [
                    _get_expert_weight_receipt(tensor[slot])
                    for tensor in module.w2_lora_a_stacked
                ],
                "w2_lora_b": [
                    _get_expert_weight_receipt(tensor[slot])
                    for tensor in module.w2_lora_b_stacked
                ],
            }
        return receipt


class _PerturbableMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def inspect_glm53_lora_parameters(self) -> dict:
        from megatron.core import parallel_state
        from megatron.core.utils import unwrap_model

        update_scope = os.environ.get("SKYRL_GLM53_UPDATE_SCOPE", "all")
        marker = (
            ".self_attention.linear_proj."
            if update_scope == "final_dense"
            else ".mlp.experts."
        )
        selected = {}
        for chunk in self.actor_module:
            model = unwrap_model(chunk)
            names = _get_final_layer_names(
                [name for name, _ in model.named_parameters()], marker
            )
            parameters = dict(model.named_parameters())
            for name in names:
                if "linear_out.weight" in name:
                    selected[name] = _get_sampled_tensor_receipt(parameters[name])
        return {
            "hostname": socket.gethostname(),
            "update_scope": update_scope,
            "rank": torch.distributed.get_rank(),
            "tensor_rank": parallel_state.get_tensor_model_parallel_rank(),
            "expert_rank": parallel_state.get_expert_model_parallel_rank(),
            "pipeline_rank": parallel_state.get_pipeline_model_parallel_rank(),
            "context_rank": parallel_state.get_context_parallel_rank(),
            "parameters": selected,
        }

    def inspect_glm53_exported_adapter(self, lora_sync_path: str) -> dict:
        from safetensors import safe_open

        rank = torch.distributed.get_rank()
        path = Path(lora_sync_path) / "adapter_model.safetensors"
        receipt = {
            "hostname": socket.gethostname(),
            "rank": rank,
            "path": str(path),
            "exists": path.exists(),
        }
        if rank != 0 or not path.exists():
            return receipt

        dtype_sizes = {"F32": 4, "BF16": 2, "F16": 2, "F64": 8}
        with safe_open(path, framework="pt", device="cpu") as adapter:
            keys = list(adapter.keys())
            schema = []
            total_bytes = 0
            for name in keys:
                tensor_slice = adapter.get_slice(name)
                shape = list(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                schema.append((name, shape, dtype))
                numel = 1
                for dimension in shape:
                    numel *= dimension
                total_bytes += numel * dtype_sizes[dtype]
            update_scope = os.environ.get("SKYRL_GLM53_UPDATE_SCOPE", "all")
            selected_names = (
                _get_final_layer_names(keys, ".self_attn.o_proj")
                if update_scope == "final_dense"
                else _get_representative_expert_names(keys)
            )
            receipt.update(
                {
                    "key_count": len(keys),
                    "total_bytes": total_bytes,
                    "schema_sha256": hashlib.sha256(
                        json.dumps(schema, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "representative_tensors": {
                        name: _get_sampled_tensor_receipt(adapter.get_tensor(name))
                        for name in selected_names
                    },
                }
            )
        return receipt

    def add_shard_symmetric_lora_b_noise(
        self, seed: int, std: float
    ) -> dict[str, float | int | str]:
        from megatron.core import parallel_state
        from megatron.core.utils import unwrap_model

        rank = torch.distributed.get_rank()
        tensor_rank = parallel_state.get_tensor_model_parallel_rank()
        expert_rank = parallel_state.get_expert_model_parallel_rank()
        pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
        context_rank = parallel_state.get_context_parallel_rank()
        updated_parameters = 0
        updated_elements = 0
        delta_norm = 0.0
        update_fingerprint = hashlib.sha256()
        update_scope = os.environ.get("SKYRL_GLM53_UPDATE_SCOPE", "all")
        assert update_scope in {
            "all",
            "expert",
            "nonexpert",
            "final_expert",
            "final_dense",
        }
        assert self._is_lora

        with torch.no_grad():
            for chunk_index, chunk in enumerate(self.actor_module):
                model = unwrap_model(chunk)
                named_parameters = list(model.named_parameters())
                local_layer_indices = [
                    int(match.group(1))
                    for name, _ in named_parameters
                    if (match := re.search(r"decoder\.layers\.(\d+)\.", name))
                ]
                final_local_layer = max(local_layer_indices, default=None)
                for name, parameter in named_parameters:
                    if not (parameter.requires_grad and "linear_out.weight" in name):
                        continue
                    is_routed_expert = ".mlp.experts." in name
                    is_expert = is_routed_expert or ".mlp.shared_experts." in name
                    if update_scope == "expert" and not is_expert:
                        continue
                    if update_scope == "nonexpert" and is_expert:
                        continue
                    if update_scope in {"final_expert", "final_dense"}:
                        layer_match = re.search(r"decoder\.layers\.(\d+)\.", name)
                        expected_module = (
                            is_routed_expert
                            if update_scope == "final_expert"
                            else ".self_attention.linear_proj." in name
                        )
                        if (
                            not expected_module
                            or parallel_state.get_pipeline_model_parallel_rank()
                            != parallel_state.get_pipeline_model_parallel_world_size()
                            - 1
                            or layer_match is None
                            or int(layer_match.group(1)) != final_local_layer
                        ):
                            continue
                    # Use the same local-coordinate update on every distributed
                    # shard. This preserves CP replicas and removes shard order
                    # as a confound in the publication check.
                    parameter_key = f"{chunk_index}:{name}"
                    digest = hashlib.sha256(parameter_key.encode()).digest()
                    parameter_seed = seed + int.from_bytes(digest[:8], "little")
                    update_fingerprint.update(
                        (
                            f"{parameter_key}:{tuple(parameter.shape)}:"
                            f"{parameter.dtype}:{parameter_seed}"
                        ).encode()
                    )
                    generator = torch.Generator(device=parameter.device)
                    generator.manual_seed(parameter_seed % (2**63 - 1))
                    noise = torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                    noise.mul_(std)
                    parameter.add_(noise)

                    updated_parameters += 1
                    updated_elements += parameter.numel()
                    delta_norm += torch.linalg.vector_norm(
                        noise, dtype=torch.float32
                    ).item()

        if update_scope in {"final_expert", "final_dense"}:
            is_final_pipeline_stage = (
                pipeline_rank
                == parallel_state.get_pipeline_model_parallel_world_size() - 1
            )
            assert (updated_parameters > 0) == is_final_pipeline_stage
        else:
            assert updated_parameters > 0, (
                "noise update found no trainable LoRA parameters"
            )
        return {
            "rank": rank,
            "tensor_rank": tensor_rank,
            "expert_rank": expert_rank,
            "pipeline_rank": pipeline_rank,
            "context_rank": context_rank,
            "update_scope": update_scope,
            "noise_std": std,
            "updated_parameters": updated_parameters,
            "updated_elements": updated_elements,
            "delta_norm": delta_norm,
            "update_fingerprint": update_fingerprint.hexdigest(),
        }


_PerturbablePolicyWorker = ray.remote(num_gpus=1)(_PerturbableMegatronPolicyWorker)


@pytest.fixture
def glm53_ray_init_fixture():
    with ray_init(
        extra_env_vars={
            "NVTE_FUSED_ATTN": "1",
            "SKYRL_GLM53_UPDATE_SCOPE": os.environ.get(
                "SKYRL_GLM53_UPDATE_SCOPE", "all"
            ),
        },
        address=os.environ.get("SKYRL_GLM53_RAY_ADDRESS"),
    ):
        yield


def _get_test_topology(model: str) -> tuple[int, int, int]:
    if model == SMALL_DRY_RUN_MODEL:
        return 1, 1, 1
    return (
        int(os.environ.get("SKYRL_GLM53_POLICY_NODES", "1")),
        POLICY_GPUS,
        INFERENCE_TP,
    )


def _get_glm53_lora_config(model: str, lora_sync_path: str) -> SkyRLTrainConfig:
    policy_nodes, policy_gpus_per_node, inference_tp = _get_test_topology(model)
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.loss_reduction = "sequence_mean"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = policy_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = policy_gpus_per_node
    cfg.trainer.policy.model.path = model
    cfg.trainer.policy.language_model_only = True
    cfg.trainer.ref.language_model_only = True
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=32,
        alpha=32,
        dropout=0.0,
        lora_sync_path=lora_sync_path,
        target_modules=(
            "all-linear"
            if model == SMALL_DRY_RUN_MODEL
            else MEGATRON_LORA_TARGET_MODULES
        ),
        max_loras=1,
    )

    megatron = cfg.trainer.policy.megatron_config
    megatron.tensor_model_parallel_size = policy_gpus_per_node
    megatron.pipeline_model_parallel_size = int(
        os.environ.get("SKYRL_GLM53_PIPELINE_PARALLEL_SIZE", "1")
    )
    megatron.context_parallel_size = int(
        os.environ.get("SKYRL_GLM53_CONTEXT_PARALLEL_SIZE", "1")
    )
    model_parallel_world_size = (
        megatron.tensor_model_parallel_size
        * megatron.pipeline_model_parallel_size
        * megatron.context_parallel_size
    )
    assert policy_nodes * policy_gpus_per_node == model_parallel_world_size, (
        "The diagnostic requires exactly one data-parallel replica: "
        f"workers={policy_nodes * policy_gpus_per_node}, "
        f"TPxPPxCP={model_parallel_world_size}"
    )
    megatron.expert_model_parallel_size = 1 if model == SMALL_DRY_RUN_MODEL else 8
    megatron.expert_tensor_parallel_size = 1
    megatron.moe_token_dispatcher_type = "alltoall"
    megatron.moe_router_load_balancing_type = "none"
    megatron.moe_grouped_gemm = True
    megatron.moe_router_score_function = "sigmoid"
    megatron.lora_config.merge_lora = False
    megatron.ddp_config.average_in_collective = False
    megatron.transformer_config_kwargs = {
        "dsa_kernel_backend": "tilelang",
        "qk_pos_emb_head_dim": 64,
        "dsa_indexer_topk_freq": 4,
        "dsa_indexer_skip_topk_offset": 3,
        "dsa_indexer_rope_interleaved": True,
        "dsa_indexer_rotate_activation": False,
        "dsa_indexer_k_norm_epsilon": 1e-6,
        "mtp_num_layers": 0,
        "mtp_use_repeated_layer": False,
        "calculate_per_token_loss": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": True,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "recompute_modules": [],
    }
    if megatron.pipeline_model_parallel_size == 2:
        # GLM's DSA top-k indices are shared in four-layer groups. Each pipeline
        # stage must therefore start on a layer that computes its own indices.
        megatron.transformer_config_kwargs["num_layers_in_first_pipeline_stage"] = 38
    if model == SMALL_DRY_RUN_MODEL:
        megatron.transformer_config_kwargs = {
            "calculate_per_token_loss": True,
            "gradient_accumulation_fusion": False,
            "sequence_parallel": False,
        }

    cfg.trainer.flash_attn = False
    cfg.trainer.remove_microbatch_padding = megatron.context_parallel_size > 1
    cfg.trainer.fused_lm_head_logprob = True
    max_sequence_length = 1024 if model == SMALL_DRY_RUN_MODEL else 32768
    cfg.trainer.max_tokens_per_microbatch = max_sequence_length
    cfg.trainer.logprobs_chunk_size = min(8192, max_sequence_length)
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1

    inference = cfg.generator.inference_engine
    inference.backend = "vllm"
    inference.fully_sharded_loras = (
        os.environ.get("SKYRL_GLM53_FULLY_SHARDED_LORAS", "0") == "1"
    )
    inference.run_engines_locally = True
    inference.language_model_only = True
    inference.num_engines = 1
    inference.tensor_parallel_size = inference_tp
    inference.pipeline_parallel_size = 1
    inference.data_parallel_size = 1
    inference.distributed_executor_backend = "ray"
    inference.weight_sync_backend = "nccl"
    inference.enforce_eager = False
    inference.gpu_memory_utilization = 0.8
    inference.max_num_seqs = 8
    inference.max_num_batched_tokens = max_sequence_length
    inference.enable_prefix_caching = False
    inference.enable_chunked_prefill = True
    inference.engine_init_kwargs = {
        "max_model_len": 32768,
        "kv_cache_dtype": "fp8",
        "disable_custom_all_reduce": True,
        "linear_backend": "triton",
        "moe_backend": "triton",
        "lora_target_modules": VLLM_LORA_TARGET_MODULES,
        "trust_remote_code": True,
    }
    if model == SMALL_DRY_RUN_MODEL:
        inference.enforce_eager = True
        inference.engine_init_kwargs = {
            "max_model_len": max_sequence_length,
            "disable_custom_all_reduce": True,
            "trust_remote_code": True,
        }
    inference.engine_init_kwargs["worker_extension_cls"] = (
        f"{__name__}._InspectableInferenceWorkerWrap"
    )

    cfg.generator.sampling_params = SamplingParams(
        max_generate_length=MAX_GENERATE_LENGTH,
        logprobs=1,
        temperature=0.0,
    )
    cfg.generator.batched = False
    cfg.generator.max_turns = 1
    validate_cfg(cfg)
    return cfg


def _get_prompt_token_ids(tokenizer) -> list[list[int]]:
    conversations = [[{"role": "user", "content": prompt}] for prompt in TEST_PROMPTS]
    return tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


async def _generate(client, tokenizer, model: str | None = None):
    prompt_token_ids = _get_prompt_token_ids(tokenizer)
    sampling_params = get_sampling_params_for_backend(
        "vllm",
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_generate_length=MAX_GENERATE_LENGTH,
            min_p=0.0,
            logprobs=1,
        ),
    )

    with Timer("generate_with_vllm"):
        output = await client.generate(
            InferenceEngineInput(
                prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
            ),
            model=model,
        )

    responses = output["response_ids"]
    rollout_logprobs = output["response_logprobs"]
    assert rollout_logprobs is not None
    response_mask, logprobs_t, training_input = _build_training_input(
        tokenizer, prompt_token_ids, responses, rollout_logprobs
    )
    return responses, response_mask, logprobs_t, training_input


async def _score_responses(client, tokenizer, responses, model):
    prompt_token_ids = _get_prompt_token_ids(tokenizer)

    async def score_response(prompt, response):
        result = await client.sample(
            {
                "json": {
                    "prompt": {"chunks": [{"tokens": prompt + response}]},
                    "num_samples": 1,
                    "sampling_params": {"temperature": 0.0, "max_tokens": 1},
                    "include_prompt_logprobs": True,
                    "model": model,
                }
            }
        )
        prompt_logprobs = result["prompt_logprobs"]
        assert prompt_logprobs is not None
        response_logprobs = prompt_logprobs[len(prompt) :]
        assert all(logprob is not None for logprob in response_logprobs)
        return response_logprobs

    with Timer("score_fixed_responses_with_vllm"):
        response_logprobs = await asyncio.gather(
            *(
                score_response(prompt, response)
                for prompt, response in zip(prompt_token_ids, responses, strict=True)
            )
        )

    return _build_training_input(
        tokenizer, prompt_token_ids, responses, response_logprobs
    )


def _build_training_input(tokenizer, prompt_token_ids, responses, rollout_logprobs):
    rewards = [[0.0] * len(response) for response in responses]
    loss_masks = [[1] * len(response) for response in responses]

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _ = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id=tokenizer.pad_token_id,
            prompts=prompt_token_ids,
            responses=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            logprobs=rollout_logprobs,
        )
    )
    assert logprobs_t is not None
    num_actions = response_mask.shape[1]
    batch_size = sequences.shape[0]
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_t,
            "loss_mask": loss_mask_t,
            "rollout_logprobs": logprobs_t,
            "rollout_expert_indices": None,
            "action_log_probs": torch.zeros(
                (batch_size, num_actions), dtype=torch.float32
            ),
            "base_action_log_probs": torch.zeros(
                (batch_size, num_actions), dtype=torch.float32
            ),
            "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
        }
    )
    training_input.metadata = {"response_length": num_actions}
    return response_mask, logprobs_t, training_input


def _get_megatron_logprobs(policy, training_input):
    refs = policy.async_run_ray_method("mesh", "forward", data=training_input)
    results = ray.get(refs)
    output = WorkerOutput.cat(policy.actor_infos, results)
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")


def _assert_logprobs_match(label, expected, actual, response_mask, threshold):
    mask = response_mask.bool()
    expected_valid = expected[mask]
    actual_valid = actual[mask]
    difference = (expected_valid - actual_valid).abs()
    mean_difference = difference.mean().item()
    print(
        f"{label}: tokens={difference.numel()}, mean_diff={mean_difference:.6f}, "
        f"max_diff={difference.max().item():.6f}"
    )
    assert torch.isfinite(difference).all()
    assert mean_difference < threshold, (
        f"{label} mean diff {mean_difference:.6f} exceeds {threshold}"
    )


def _assert_logprobs_changed(before, after, response_mask):
    difference = (before[response_mask.bool()] - after[response_mask.bool()]).abs()
    mean_difference = difference.mean().item()
    print(
        f"dummy LoRA update effect: tokens={difference.numel()}, "
        f"mean_diff={mean_difference:.6f}, max_diff={difference.max().item():.6f}"
    )
    assert torch.isfinite(difference).all()
    assert mean_difference > MIN_UPDATED_LOGPROB_DIFF, (
        f"dummy LoRA update changed mean logprob by only {mean_difference:.6f}"
    )


def _init_perturbable_policy(cfg, policy_nodes, policy_gpus_per_node):
    original_policy_worker = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _PerturbablePolicyWorker
    try:
        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_nodes=policy_nodes,
            num_gpus_per_node=policy_gpus_per_node,
            cfg=cfg,
        )
    finally:
        _megatron_worker_mod.PolicyWorker = original_policy_worker

    return policy


async def _create_inference_engine(stack, cfg, model):
    with Timer("initialize_vllm"):
        state = await asyncio.to_thread(
            InferenceEngineState.create,
            cfg=cfg,
            model=model,
            use_local=True,
            colocate_all=False,
            backend="vllm",
            enable_lora=True,
        )
    return await stack.enter_async_context(state)


async def _create_policy(cfg, policy_nodes, policy_gpus_per_node):
    with Timer("initialize_megatron"):
        return await asyncio.to_thread(
            _init_perturbable_policy, cfg, policy_nodes, policy_gpus_per_node
        )


def _print_boundary_receipt(label: str, receipt) -> None:
    print(f"GLM53_LORA_BOUNDARY {label} {json.dumps(receipt, sort_keys=True)}")


def _inspect_policy_boundary(policy, method: str, *args):
    return ray.get(policy.async_run_ray_method("pass_through", method, *args))


async def _inspect_vllm_boundary(client):
    return await client._call_all_servers(
        "/collective_rpc",
        {"method": "inspect_glm53_lora_buffers", "kwargs": {}},
    )


@pytest.mark.asyncio
@pytest.mark.megatron
@pytest.mark.b300
async def test_glm53_lora_init_and_dummy_update_match_vllm(glm53_ray_init_fixture):
    model = os.environ.get("SKYRL_GLM53_MODEL", MODEL)
    update_scope = os.environ.get("SKYRL_GLM53_UPDATE_SCOPE", "all")
    policy_nodes, policy_gpus_per_node, inference_tp = _get_test_topology(model)
    policy_gpus = policy_nodes * policy_gpus_per_node
    shared_dir = Path(os.environ["SKYRL_GLM53_SHARED_DIR"])
    lora_sync_path = shared_dir / f"glm53-lora-parity-{uuid.uuid4().hex}"
    lora_sync_path.mkdir(parents=True)

    assert ray.cluster_resources().get("GPU", 0) >= policy_gpus + inference_tp, (
        f"LoRA parity requires {policy_gpus + inference_tp} GPUs in the connected Ray cluster"
    )

    cfg = _get_glm53_lora_config(model, str(lora_sync_path))
    tokenizer = get_tokenizer(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    completed = False
    try:
        async with AsyncExitStack() as stack:
            with Timer("initialize_vllm_and_megatron_concurrently"):
                init_results = await asyncio.gather(
                    _create_inference_engine(stack, cfg, model),
                    _create_policy(cfg, policy_nodes, policy_gpus_per_node),
                    return_exceptions=True,
                )
            for result in init_results:
                if isinstance(result, BaseException):
                    raise result
            engines, policy = init_results
            client = engines.client
            adapter_loaded = False
            try:
                base_responses, base_mask, base_logprobs, base_input = await _generate(
                    client, tokenizer, model
                )
                base_mask, base_logprobs, base_input = await _score_responses(
                    client, tokenizer, base_responses, model
                )

                with Timer("initialize_weight_sync"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "init_weight_sync_state",
                            client,
                            cfg.generator.inference_engine,
                        )
                    )

                initial_megatron_logprobs = _get_megatron_logprobs(policy, base_input)
                _assert_logprobs_match(
                    "base vLLM vs initialized Megatron LoRA",
                    base_logprobs,
                    initial_megatron_logprobs,
                    base_mask,
                    MEGATRON_MEAN_DIFF_THRESHOLD,
                )
                trainer_initial_receipts = _inspect_policy_boundary(
                    policy, "inspect_glm53_lora_parameters"
                )
                assert {
                    receipt["update_scope"] for receipt in trainer_initial_receipts
                } == {update_scope}
                _print_boundary_receipt("trainer_initial", trainer_initial_receipts)

                with Timer("publish_initialized_lora"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "broadcast_to_inference_engines",
                            client,
                            cfg.generator.inference_engine,
                        )
                    )
                adapter_loaded = True
                await client.reset_prefix_cache()
                _print_boundary_receipt(
                    "export_initial",
                    _inspect_policy_boundary(
                        policy,
                        "inspect_glm53_exported_adapter",
                        str(lora_sync_path),
                    ),
                )
                vllm_initial_receipt = await _inspect_vllm_boundary(client)
                assert {
                    worker_receipt["update_scope"]
                    for server_receipt in vllm_initial_receipt.values()
                    for worker_receipt in server_receipt["body"]["results"]
                } == {update_scope}
                if update_scope == "final_dense":
                    expected_fully_sharded = (
                        os.environ.get("SKYRL_GLM53_FULLY_SHARDED_LORAS", "0") == "1"
                    )
                    module_types = {
                        module_receipt["module_type"]
                        for server_receipt in vllm_initial_receipt.values()
                        for worker_receipt in server_receipt["body"]["results"]
                        for module_receipt in worker_receipt["kernel_buffers"].values()
                    }
                    assert {
                        "ShardedLoRA" in module_type for module_type in module_types
                    } == {expected_fully_sharded}
                _print_boundary_receipt("vllm_initial", vllm_initial_receipt)

                adapter_name = resolve_policy_model_name(cfg)
                lora_mask, lora_logprobs, lora_input = await _score_responses(
                    client, tokenizer, base_responses, adapter_name
                )
                assert torch.equal(base_mask, lora_mask)
                _assert_logprobs_match(
                    "base vLLM vs initialized vLLM LoRA",
                    base_logprobs,
                    lora_logprobs,
                    base_mask,
                    VLLM_MEAN_DIFF_THRESHOLD,
                )

                lora_megatron_logprobs = _get_megatron_logprobs(policy, lora_input)
                _assert_logprobs_match(
                    "initialized vLLM LoRA vs Megatron LoRA",
                    lora_logprobs,
                    lora_megatron_logprobs,
                    lora_mask,
                    MEGATRON_MEAN_DIFF_THRESHOLD,
                )

                noise_std = (
                    FINAL_DENSE_LORA_NOISE_STD
                    if update_scope == "final_dense"
                    else LORA_NOISE_STD
                )
                with Timer("apply_dummy_lora_update"):
                    update_receipts = ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "add_shard_symmetric_lora_b_noise",
                            LORA_NOISE_SEED,
                            noise_std,
                        )
                    )
                for receipt in update_receipts:
                    assert receipt["update_scope"] == update_scope
                    should_update = (
                        update_scope not in {"final_expert", "final_dense"}
                        or receipt["pipeline_rank"]
                        == cfg.trainer.policy.megatron_config.pipeline_model_parallel_size
                        - 1
                    )
                    assert (receipt["updated_parameters"] > 0) == should_update
                    assert (receipt["updated_elements"] > 0) == should_update
                    assert (receipt["delta_norm"] > 0) == should_update
                assert {receipt["rank"] for receipt in update_receipts} == set(
                    range(policy_gpus)
                )
                receipts_by_pipeline_stage = {}
                for receipt in update_receipts:
                    receipts_by_pipeline_stage.setdefault(
                        receipt["pipeline_rank"], []
                    ).append(receipt)
                for stage_receipts in receipts_by_pipeline_stage.values():
                    assert (
                        len(
                            {
                                (
                                    receipt["updated_parameters"],
                                    receipt["updated_elements"],
                                    receipt["delta_norm"],
                                    receipt["update_fingerprint"],
                                )
                                for receipt in stage_receipts
                            }
                        )
                        == 1
                    ), "LoRA perturbation must preserve distributed replicas"
                receipts_by_model_coordinate = {}
                for receipt in update_receipts:
                    coordinate = (
                        receipt["tensor_rank"],
                        receipt["expert_rank"],
                        receipt["pipeline_rank"],
                    )
                    receipts_by_model_coordinate.setdefault(coordinate, []).append(
                        receipt
                    )
                for replicas in receipts_by_model_coordinate.values():
                    assert (
                        len(replicas)
                        == cfg.trainer.policy.megatron_config.context_parallel_size
                    )
                    assert {
                        (
                            receipt["updated_parameters"],
                            receipt["updated_elements"],
                            receipt["delta_norm"],
                        )
                        for receipt in replicas
                    } == {
                        (
                            replicas[0]["updated_parameters"],
                            replicas[0]["updated_elements"],
                            replicas[0]["delta_norm"],
                        )
                    }

                updated_logprobs = _get_megatron_logprobs(policy, lora_input)
                _print_boundary_receipt(
                    "trainer_updated",
                    _inspect_policy_boundary(policy, "inspect_glm53_lora_parameters"),
                )
                _assert_logprobs_changed(
                    lora_megatron_logprobs, updated_logprobs, lora_mask
                )

                await client.unload_lora_adapter(adapter_name)
                adapter_loaded = False
                with Timer("publish_dummy_updated_lora"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "broadcast_to_inference_engines",
                            client,
                            cfg.generator.inference_engine,
                        )
                    )
                adapter_loaded = True
                await client.reset_prefix_cache()
                _print_boundary_receipt(
                    "export_updated",
                    _inspect_policy_boundary(
                        policy,
                        "inspect_glm53_exported_adapter",
                        str(lora_sync_path),
                    ),
                )
                _print_boundary_receipt(
                    "vllm_updated", await _inspect_vllm_boundary(client)
                )

                (
                    updated_mask,
                    updated_vllm_logprobs,
                    updated_input,
                ) = await _score_responses(
                    client, tokenizer, base_responses, adapter_name
                )
                updated_megatron_logprobs = _get_megatron_logprobs(
                    policy, updated_input
                )
                valid = updated_mask.bool()
                sampler_delta = updated_vllm_logprobs[valid] - lora_logprobs[valid]
                trainer_delta = (
                    updated_megatron_logprobs[valid] - lora_megatron_logprobs[valid]
                )
                delta_error = (sampler_delta - trainer_delta).abs()
                cosine = torch.nn.functional.cosine_similarity(
                    sampler_delta.float().unsqueeze(0),
                    trainer_delta.float().unsqueeze(0),
                ).item()
                scale = (
                    torch.dot(sampler_delta.float(), trainer_delta.float())
                    / torch.dot(trainer_delta.float(), trainer_delta.float())
                ).item()
                mean_error = delta_error.mean().item()
                trainer_mean = trainer_delta.abs().mean().item()
                relative_mean_error = mean_error / trainer_mean
                print(
                    "direct update delta parity: "
                    f"tokens={delta_error.numel()}, "
                    f"mean_diff={mean_error:.6f}, "
                    f"p99_diff={torch.quantile(delta_error.float(), 0.99).item():.6f}, "
                    f"max_diff={delta_error.max().item():.6f}, "
                    f"sampler_mean={sampler_delta.abs().mean().item():.6f}, "
                    f"trainer_mean={trainer_mean:.6f}, "
                    f"relative_mean_error={relative_mean_error:.6f}, "
                    f"cosine={cosine:.6f}, scale={scale:.6f}"
                )
                minimum_trainer_mean = (
                    1e-2 if model == SMALL_DRY_RUN_MODEL else MIN_DIRECT_UPDATE_MEAN
                )
                assert trainer_mean > minimum_trainer_mean
                assert cosine > MIN_DIRECT_UPDATE_COSINE
                assert MIN_DIRECT_UPDATE_SCALE < scale < MAX_DIRECT_UPDATE_SCALE
                assert relative_mean_error < MAX_DIRECT_UPDATE_RELATIVE_MEAN_ERROR
                assert mean_error < 0.075
                assert torch.quantile(delta_error.float(), 0.99).item() < 0.75
                assert delta_error.max().item() < 5.0
                _assert_logprobs_match(
                    "dummy-updated vLLM LoRA vs Megatron LoRA",
                    updated_vllm_logprobs,
                    updated_megatron_logprobs,
                    updated_mask,
                    MEGATRON_MEAN_DIFF_THRESHOLD,
                )
            finally:
                if adapter_loaded:
                    await client.unload_lora_adapter(resolve_policy_model_name(cfg))
        completed = True
    finally:
        if completed:
            shutil.rmtree(lora_sync_path)
        else:
            print(f"preserved failed adapter evidence at {lora_sync_path}")
