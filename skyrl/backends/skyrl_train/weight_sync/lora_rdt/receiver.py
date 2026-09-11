"""Pull rank-owned LoRA slices through NIXL and stage a vLLM adapter."""

from typing import Any, Mapping

import ray
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_layout import (
    convert_moe_experts_lora_to_vllm,
)

from .bridge_sources import LoRABridgeSourceLayout
from .consumer_plan import LoRAConsumerPlan, LoRAConsumerPull
from .contracts import LoRAAdapterLayout, LoRAUpdateRequest
from .vllm_adapter import (
    build_vllm_lora_model,
    get_vllm_local_lora_plan,
    stage_vllm_local_lora_factors,
    stage_vllm_lora_model,
)


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
    validate_lora_request_layout(layout, request)
    names_by_source: dict[int, list[str]] = {}
    expected = {}
    for tensor in layout.tensors:
        if tensor.destination_rank != inference_rank:
            continue
        names_by_source.setdefault(tensor.source_rank, []).append(tensor.key)
        expected[tensor.key] = tensor
    if not expected:
        raise ValueError(f"LoRA layout has no tensors for inference rank {inference_rank}")
    missing_producers = sorted(set(names_by_source) - set(producers))
    if missing_producers:
        raise ValueError(f"LoRA layout requires unavailable producer ranks {missing_producers}")
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
                raise ValueError(f"LoRA tensor {name!r} was returned by more than one producer")
            source_tensors[name] = tensor
    if set(source_tensors) != set(expected):
        raise ValueError(f"LoRA pull returned {sorted(source_tensors)}, expected {sorted(expected)}")
    for name, tensor in source_tensors.items():
        spec = expected[name]
        if tensor.dtype is not torch.float32:
            raise ValueError(f"LoRA pull returned {name!r} with dtype {tensor.dtype}, expected float32")
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(f"LoRA pull returned {name!r} with shape {tuple(tensor.shape)}, expected {spec.shape}")
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


def acknowledge_lora_generation(producers: Mapping[int, Any], generation: int, consumer_id: int) -> list[bool]:
    """Notify every contributing producer after global activation succeeds."""
    return ray.get([producer.acknowledge.remote(generation, consumer_id) for producer in producers.values()])


def validate_lora_request_layout(
    layout: LoRAAdapterLayout | LoRABridgeSourceLayout, request: LoRAUpdateRequest
) -> None:
    if request.adapter_name != layout.adapter_name:
        raise ValueError(f"LoRA request adapter {request.adapter_name!r} does not match {layout.adapter_name!r}")
    if request.layout_digest != layout.layout_digest:
        raise ValueError("LoRA request layout digest does not match receiver layout")
    if request.source_dtype != layout.source_dtype:
        raise ValueError("LoRA request source dtype does not match receiver layout")


def resolve_lora_rdt_producers(
    actor_names_by_rank: Mapping[int, str],
    namespace: str | None,
) -> dict[int, Any]:
    """Resolve the validated named NIXL producer actors for one update."""
    return {
        source_rank: ray.get_actor(actor_name, namespace=namespace)
        for source_rank, actor_name in actor_names_by_rank.items()
    }


def pull_reconstruct_and_stage_lora_adapter(
    producers: Mapping[int, Any],
    layout: "LoRABridgeSourceLayout",
    request: LoRAUpdateRequest,
    adapter_id: int,
    adapter_config: Mapping[str, Any],
    model_runner: Any,
    device: str,
    weights_mapper: Any = None,
    skip_prefixes: list[str] | None = None,
    model_vocab_size: int | None = None,
) -> Any:
    """NIXL-pull Bridge sources, reconstruct PEFT tensors, and stage a BF16 adapter."""
    from .bridge_sources import reconstruct_lora_bridge_tensors

    if request.adapter_name != layout.adapter_name:
        raise ValueError("LoRA request adapter does not match Bridge source layout")
    if request.layout_digest != layout.layout_digest:
        raise ValueError("LoRA request layout digest does not match Bridge source layout")
    if request.source_dtype != layout.source_dtype:
        raise ValueError("LoRA request source dtype does not match Bridge source layout")
    names_by_source: dict[int, list[str]] = {}
    for source in layout.sources:
        names_by_source.setdefault(source.source_rank, []).append(source.key)
    if not names_by_source:
        raise ValueError("lora_rdt requires at least one Bridge adapter source")
    missing_producers = sorted(set(names_by_source) - set(producers))
    if missing_producers:
        raise ValueError(f"LoRA Bridge sources require unavailable producer ranks {missing_producers}")
    pulled_groups = ray.get(
        [
            producers[source_rank].pull.remote(request.generation, sorted(set(names)))
            for source_rank, names in names_by_source.items()
        ]
    )
    pulled_by_source = dict(zip(names_by_source, pulled_groups, strict=True))
    tensors: dict[tuple[str, int, int], torch.Tensor] = {}
    for source in layout.sources:
        source_tensors = pulled_by_source[source.source_rank]
        tensor = source_tensors.get(source.key)
        if tensor is None:
            raise ValueError(f"LoRA Bridge source rank {source.source_rank} did not return {source.key!r}")
        key = (source.key, source.tensor_parallel_rank, source.expert_parallel_rank)
        if key in tensors:
            raise ValueError(f"LoRA Bridge pull returned duplicate shard {key!r}")
        tensors[key] = tensor
    bridge_tensors = reconstruct_lora_bridge_tensors(layout.sources, tensors)
    peft_tensors = convert_moe_experts_lora_to_vllm(
        {
            name if name.startswith("base_model.model.") else f"base_model.model.{name}": tensor
            for name, tensor in bridge_tensors.items()
        }
    )
    model = build_vllm_lora_model(
        adapter_id=adapter_id,
        adapter_config=adapter_config,
        source_tensors=peft_tensors,
        device=device,
        dtype=torch.bfloat16,
        model_vocab_size=model_vocab_size,
        weights_mapper=weights_mapper,
        skip_prefixes=skip_prefixes,
    )
    stage_vllm_lora_model(model_runner, model)
    return model


def pull_and_stage_local_lora_adapter(
    producers: Mapping[int, Any],
    layout: LoRABridgeSourceLayout,
    request: LoRAUpdateRequest,
    consumer_plan: LoRAConsumerPlan,
    adapter_id: int,
    model_runner: Any,
    device: str,
) -> None:
    """Pull only this rank's source rectangles and register local BF16 factors."""
    from .consumer_plan import assemble_lora_consumer_factors

    validate_lora_request_layout(layout, request)
    if consumer_plan.source_layout_digest != layout.layout_digest:
        raise ValueError("LoRA consumer plan does not match the source layout")
    pulls_by_source: dict[int, list[LoRAConsumerPull]] = {}
    for pull in consumer_plan.pulls:
        pulls_by_source.setdefault(pull.source_rank, []).append(pull)
    missing_producers = sorted(set(pulls_by_source) - set(producers))
    if missing_producers:
        raise ValueError(f"LoRA consumer plan requires unavailable producer ranks {missing_producers}")
    groups = ray.get(
        [
            producers[rank].pull_slices.remote(request.generation, [pull.source_slice for pull in pulls])
            for rank, pulls in pulls_by_source.items()
        ]
    )
    pulled = {}
    for pulls, tensors in zip(pulls_by_source.values(), groups, strict=True):
        if len(tensors) != len(pulls):
            raise ValueError("LoRA producer returned an incorrect number of source slices")
        pulled.update(zip(pulls, tensors, strict=True))
    factors = assemble_lora_consumer_factors(consumer_plan, pulled, device)
    stage_vllm_local_lora_factors(model_runner, adapter_id, consumer_plan.receiver_plan, factors)
    # Keep Ray receive buffers alive until all dependent CUDA copies complete.
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def build_vllm_lora_consumer_plan(
    layout: LoRABridgeSourceLayout, adapter_config: Mapping[str, Any], model_runner: Any
) -> LoRAConsumerPlan:
    """Build one fixed source-to-local-buffer plan for a named adapter."""
    from .consumer_plan import build_lora_consumer_plan

    return build_lora_consumer_plan(layout, get_vllm_local_lora_plan(model_runner, adapter_config))
