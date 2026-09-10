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
