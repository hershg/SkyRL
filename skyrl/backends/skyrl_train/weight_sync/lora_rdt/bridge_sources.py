"""Normalize Megatron-Bridge rank-local adapter records for LoRA RDT."""

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import torch


@dataclass(frozen=True)
class LoRABridgeSource:
    """Content-independent metadata for one rank-local Bridge adapter source."""

    key: str
    hf_param_names: tuple[str, ...]
    component: Literal["linear_in", "linear_out"]
    transform: Literal[
        "identity", "replicate", "split_qkv", "split_gated_mlp", "split_gdn_in_proj"
    ]
    shape: tuple[int, ...]
    tensor_parallel_axis: int | None
    tensor_parallel_rank: int
    tensor_parallel_size: int
    expert_parallel_axis: int | None
    expert_parallel_rank: int
    expert_parallel_size: int
    transform_config: tuple[tuple[str, int | bool | None], ...]


def extract_lora_bridge_sources(
    records: Iterable[Any],
) -> tuple[dict[str, torch.Tensor], tuple[LoRABridgeSource, ...]]:
    """Detach tensors from Bridge records and return stable transport metadata.

    The Bridge extension supplies a local FP32 snapshot per source tensor. This
    function keeps that storage separate from its serializable layout metadata so
    a producer can expose only the tensors over NIXL and publish the metadata on
    the control plane.
    """
    tensors: dict[str, torch.Tensor] = {}
    sources: list[LoRABridgeSource] = []
    for record in records:
        key = record.global_param_name
        if key in tensors:
            raise ValueError(f"Bridge adapter records contain duplicate source {key!r}")
        tensor = record.weight
        if tensor.dtype is not torch.float32:
            raise ValueError(
                f"lora_rdt requires float32 Bridge source {key!r}, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"lora_rdt requires contiguous Bridge source {key!r}")
        if not record.hf_param_names:
            raise ValueError(f"Bridge adapter source {key!r} has no HF parameter names")
        tensors[key] = tensor
        sources.append(
            LoRABridgeSource(
                key=key,
                hf_param_names=tuple(record.hf_param_names),
                component=record.component,
                transform=record.transform,
                shape=tuple(tensor.shape),
                tensor_parallel_axis=record.tensor_parallel_axis,
                tensor_parallel_rank=record.tensor_parallel_rank,
                tensor_parallel_size=record.tensor_parallel_size,
                expert_parallel_axis=record.expert_parallel_axis,
                expert_parallel_rank=record.expert_parallel_rank,
                expert_parallel_size=record.expert_parallel_size,
                transform_config=tuple(record.transform_config),
            )
        )
    if not sources:
        raise ValueError("lora_rdt requires at least one Bridge adapter source")
    sources.sort(key=lambda source: source.key)
    return tensors, tuple(sources)


def validate_lora_bridge_source_layout(
    sources: Iterable[LoRABridgeSource],
) -> Mapping[str, LoRABridgeSource]:
    """Validate a fixed Bridge source layout before publishing any generation."""
    source_tuple = tuple(sources)
    layout = {source.key: source for source in source_tuple}
    if not layout:
        raise ValueError("lora_rdt requires at least one Bridge adapter source")
    if len(layout) != len(source_tuple):
        raise ValueError("lora_rdt Bridge source keys must be unique")
    for source in layout.values():
        if any(dimension <= 0 for dimension in source.shape):
            raise ValueError(
                f"Bridge source {source.key!r} has invalid shape {source.shape!r}"
            )
        if source.tensor_parallel_size <= 0 or source.expert_parallel_size <= 0:
            raise ValueError(f"Bridge source {source.key!r} has invalid parallel sizes")
        if not 0 <= source.tensor_parallel_rank < source.tensor_parallel_size:
            raise ValueError(
                f"Bridge source {source.key!r} has invalid tensor-parallel rank"
            )
        if not 0 <= source.expert_parallel_rank < source.expert_parallel_size:
            raise ValueError(
                f"Bridge source {source.key!r} has invalid expert-parallel rank"
            )
    return layout


def reconstruct_lora_bridge_tensors(
    sources: Iterable[LoRABridgeSource],
    tensors: Mapping[tuple[str, int, int], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct PEFT tensors from pulled rank-local Bridge source shards.

    ``tensors`` keys are ``(global_param_name, tp_rank, ep_rank)``. The caller
    obtains those tensors through NIXL; this function only performs deterministic
    local assembly. Fused QKV and GDN sources are intentionally rejected here
    because their conversion needs the exact Megatron transformer configuration.
    """
    grouped: dict[str, list[LoRABridgeSource]] = {}
    for source in sources:
        grouped.setdefault(source.key, []).append(source)
    result: dict[str, torch.Tensor] = {}
    for key, group in grouped.items():
        first = group[0]
        if any(
            source.hf_param_names != first.hf_param_names
            or source.component != first.component
            or source.transform != first.transform
            or source.tensor_parallel_axis != first.tensor_parallel_axis
            or source.expert_parallel_axis != first.expert_parallel_axis
            or source.transform_config != first.transform_config
            for source in group
        ):
            raise ValueError(f"Bridge source {key!r} has inconsistent shard metadata")
        local_by_ep: list[torch.Tensor] = []
        for ep_rank in range(first.expert_parallel_size):
            ep_sources = [
                source for source in group if source.expert_parallel_rank == ep_rank
            ]
            if not ep_sources:
                raise ValueError(f"Bridge source {key!r} is missing EP rank {ep_rank}")
            shards = []
            expected_tp_ranks = range(first.tensor_parallel_size)
            for tp_rank in expected_tp_ranks:
                matching = [
                    source
                    for source in ep_sources
                    if source.tensor_parallel_rank == tp_rank
                ]
                if len(matching) != 1:
                    raise ValueError(
                        f"Bridge source {key!r} has invalid TP ownership for rank {tp_rank}"
                    )
                source = matching[0]
                tensor_key = (key, tp_rank, ep_rank)
                tensor = tensors.get(tensor_key)
                if tensor is None:
                    raise ValueError(
                        f"Bridge source {key!r} is missing pulled tensor {tensor_key!r}"
                    )
                if tensor.dtype is not torch.float32:
                    raise ValueError(
                        f"lora_rdt requires float32 Bridge source {key!r}, got {tensor.dtype}"
                    )
                if tuple(tensor.shape) != source.shape:
                    raise ValueError(
                        f"Bridge source {key!r} tensor {tensor_key!r} has shape {tuple(tensor.shape)}, "
                        f"expected {source.shape}"
                    )
                shards.append(tensor)
            if first.tensor_parallel_axis is None:
                local_by_ep.append(shards[0])
            else:
                local_by_ep.append(torch.cat(shards, dim=first.tensor_parallel_axis))
        if first.expert_parallel_axis is None:
            assembled = local_by_ep[0]
        else:
            assembled = torch.cat(local_by_ep, dim=first.expert_parallel_axis)
        _emit_reconstructed_lora_tensors(result, first, assembled)
    return result


def _emit_reconstructed_lora_tensors(
    result: dict[str, torch.Tensor],
    source: LoRABridgeSource,
    tensor: torch.Tensor,
) -> None:
    """Apply a Bridge-declared post-assembly transform to one source tensor."""
    if source.transform == "identity":
        if len(source.hf_param_names) != 1:
            raise ValueError(
                f"Bridge source {source.key!r} identity transform requires one HF name"
            )
        result[source.hf_param_names[0]] = tensor
        return
    if source.transform == "replicate":
        for name in source.hf_param_names:
            result[name] = tensor
        return
    if source.transform == "split_gated_mlp":
        if len(source.hf_param_names) != 2:
            raise ValueError(
                f"Bridge source {source.key!r} gated transform requires two HF names"
            )
        gate, up = torch.chunk(tensor, 2, dim=0)
        result[source.hf_param_names[0]] = gate
        result[source.hf_param_names[1]] = up
        return
    if source.transform == "split_qkv":
        if len(source.hf_param_names) != 3:
            raise ValueError(
                f"Bridge source {source.key!r} QKV transform requires three HF names"
            )
        q, k, v = _split_qkv_lora_tensor(tensor, dict(source.transform_config))
        result[source.hf_param_names[0]] = q
        result[source.hf_param_names[1]] = k
        result[source.hf_param_names[2]] = v
        return
    raise ValueError(
        f"Bridge source {source.key!r} requires {source.transform!r} conversion with the Megatron config"
    )


def _split_qkv_lora_tensor(
    tensor: torch.Tensor,
    config: Mapping[str, int | bool | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a full Megatron interleaved QKV LoRA-B tensor into PEFT tensors."""
    required = ("num_attention_heads", "num_query_groups", "kv_channels", "hidden_size")
    if any(config.get(field) is None for field in required):
        raise ValueError("QKV LoRA source is missing its Bridge transform config")
    num_heads = int(config["num_attention_heads"])
    num_groups = int(config["num_query_groups"])
    head_size = int(config["kv_channels"] or int(config["hidden_size"]) // num_heads)
    heads_per_group = num_heads // num_groups
    attention_output_gate = bool(config.get("attention_output_gate", False))
    total_heads_per_group = (
        2 * heads_per_group + 2 if attention_output_gate else heads_per_group + 2
    )
    qkv_total_dim = (
        2 * num_heads + 2 * num_groups
        if attention_output_gate
        else num_heads + 2 * num_groups
    )
    if tensor.ndim != 2 or tensor.shape[0] != qkv_total_dim * head_size:
        raise ValueError(
            f"QKV LoRA source has shape {tuple(tensor.shape)}, expected first dimension "
            f"{qkv_total_dim * head_size}"
        )
    feature_dim = tensor.shape[1]
    qkv = tensor.view(qkv_total_dim, head_size, feature_dim)
    q_indices = torch.cat(
        [
            torch.arange(
                total_heads_per_group * index,
                total_heads_per_group * index + heads_per_group,
            )
            for index in range(num_groups)
        ]
    )
    k_indices = torch.arange(
        total_heads_per_group - 2, qkv_total_dim, total_heads_per_group
    )
    v_indices = torch.arange(
        total_heads_per_group - 1, qkv_total_dim, total_heads_per_group
    )
    q = qkv[q_indices]
    if attention_output_gate:
        z_indices = torch.cat(
            [
                torch.arange(
                    total_heads_per_group * index + heads_per_group,
                    total_heads_per_group * index + heads_per_group * 2,
                )
                for index in range(num_groups)
            ]
        )
        q = torch.cat([q, qkv[z_indices]], dim=1)
    return (
        q.reshape(-1, feature_dim),
        qkv[k_indices].reshape(-1, feature_dim),
        qkv[v_indices].reshape(-1, feature_dim),
    )
