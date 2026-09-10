"""Pull rank-owned LoRA slices through NIXL and stage a vLLM adapter."""

from typing import Any, Mapping

import ray
import torch

from .contracts import LoRAAdapterLayout, LoRAUpdateRequest
from .vllm_adapter import build_vllm_lora_model, stage_vllm_lora_model


def pull_and_stage_lora_adapter(
    producers: Mapping[int, Any],
    layout: LoRAAdapterLayout,
    request: LoRAUpdateRequest,
    inference_rank: int,
    adapter_id: int,
    adapter_config: Mapping[str, Any],
    model_runner: Any,
    device: str,
    weights_mapper: Any = None,
    skip_prefixes: list[str] | None = None,
    model_vocab_size: int | None = None,
) -> Any:
    """NIXL-pull one inference rank's slices and stage its independent BF16 adapter."""
    _validate_request(layout, request)
    names_by_source: dict[int, list[str]] = {}
    expected = {}
    for tensor in layout.tensors:
        if tensor.destination_rank != inference_rank:
            continue
        names_by_source.setdefault(tensor.source_rank, []).append(tensor.key)
        expected[tensor.key] = tensor
    if not expected:
        raise ValueError(
            f"LoRA layout has no tensors for inference rank {inference_rank}"
        )
    missing_producers = sorted(set(names_by_source) - set(producers))
    if missing_producers:
        raise ValueError(
            f"LoRA layout requires unavailable producer ranks {missing_producers}"
        )
    pulled_groups = ray.get(
        [
            producers[source_rank].pull.remote(request.generation, names)
            for source_rank, names in names_by_source.items()
        ]
    )
    source_tensors: dict[str, torch.Tensor] = {}
    for tensors in pulled_groups:
        for name, tensor in tensors.items():
            if name in source_tensors:
                raise ValueError(
                    f"LoRA tensor {name!r} was returned by more than one producer"
                )
            source_tensors[name] = tensor
    if set(source_tensors) != set(expected):
        raise ValueError(
            f"LoRA pull returned {sorted(source_tensors)}, expected {sorted(expected)}"
        )
    for name, tensor in source_tensors.items():
        spec = expected[name]
        if tensor.dtype is not torch.float32:
            raise ValueError(
                f"LoRA pull returned {name!r} with dtype {tensor.dtype}, expected float32"
            )
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(
                f"LoRA pull returned {name!r} with shape {tuple(tensor.shape)}, expected {spec.shape}"
            )
    model = build_vllm_lora_model(
        adapter_id=adapter_id,
        adapter_config=adapter_config,
        source_tensors=source_tensors,
        device=device,
        dtype=torch.bfloat16,
        model_vocab_size=model_vocab_size,
        weights_mapper=weights_mapper,
        skip_prefixes=skip_prefixes,
    )
    stage_vllm_lora_model(model_runner, model)
    return model


def acknowledge_lora_generation(
    producers: Mapping[int, Any], generation: int, consumer_id: int
) -> list[bool]:
    """Notify every contributing producer after global activation succeeds."""
    return ray.get(
        [
            producer.acknowledge.remote(generation, consumer_id)
            for producer in producers.values()
        ]
    )


def _validate_request(layout: LoRAAdapterLayout, request: LoRAUpdateRequest) -> None:
    if request.adapter_name != layout.adapter_name:
        raise ValueError(
            f"LoRA request adapter {request.adapter_name!r} does not match {layout.adapter_name!r}"
        )
    if request.layout_digest != layout.layout_digest:
        raise ValueError("LoRA request layout digest does not match receiver layout")
    if request.source_dtype != layout.source_dtype:
        raise ValueError("LoRA request source dtype does not match receiver layout")
