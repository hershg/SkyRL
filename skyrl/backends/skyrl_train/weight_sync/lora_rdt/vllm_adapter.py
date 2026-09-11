"""Build vLLM adapter objects from a completed LoRA RDT receive."""

from typing import Any, Mapping

import torch


def build_vllm_lora_model(
    adapter_id: int,
    adapter_config: Mapping[str, Any],
    source_tensors: Mapping[str, torch.Tensor],
    device: str,
    dtype: torch.dtype,
    model_vocab_size: int | None = None,
    weights_mapper: Any = None,
    skip_prefixes: list[str] | None = None,
) -> Any:
    """Construct a vLLM LoRAModel with independent destination tensor storage."""
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    if adapter_id <= 0:
        raise ValueError(f"LoRA adapter ids must be positive, got {adapter_id}")
    if dtype is not torch.bfloat16:
        raise ValueError(f"lora_rdt requires bfloat16 inference buffers, got {dtype}")
    for name, tensor in source_tensors.items():
        if tensor.dtype is not torch.float32:
            raise ValueError(f"lora_rdt requires float32 source tensor {name!r}, got {tensor.dtype}")
    peft_helper = PEFTHelper.from_dict(dict(adapter_config))
    return LoRAModel.from_lora_tensors(
        adapter_id,
        {name: tensor.to(device=device, dtype=dtype) for name, tensor in source_tensors.items()},
        peft_helper,
        device=device,
        dtype=dtype,
        model_vocab_size=model_vocab_size,
        weights_mapper=weights_mapper,
        skip_prefixes=skip_prefixes,
    )


def stage_vllm_lora_model(model_runner: Any, lora_model: Any) -> None:
    """Register a new adapter id without changing the active vLLM slots."""
    manager = _get_vllm_lora_manager(model_runner)
    adapter_manager = manager._adapter_manager
    _validate_staging_capacity(manager, lora_model.id)
    if not adapter_manager.add_adapter(lora_model):
        raise RuntimeError(f"vLLM declined LoRA adapter id {lora_model.id}")


def activate_staged_vllm_lora_model(model_runner: Any, adapter_id: int) -> None:
    """Activate a previously staged adapter id in an unused vLLM LoRA slot."""
    manager = _get_vllm_lora_manager(model_runner)
    if adapter_id not in manager.list_adapters():
        raise ValueError(f"LoRA adapter id {adapter_id} was not staged")
    # vLLM returns False when this adapter is already active, including rollback.
    manager._adapter_manager.activate_adapter(adapter_id)


def discard_staged_vllm_lora_model(model_runner: Any, adapter_id: int) -> None:
    """Release a staged adapter id after a failed generation."""
    manager = _get_vllm_lora_manager(model_runner)
    manager.remove_adapter(adapter_id)


def _get_vllm_lora_manager(model_runner: Any) -> Any:
    manager = getattr(model_runner, "lora_manager", None)
    if manager is None:
        raise RuntimeError("lora_rdt requires a vLLM model runner with LoRA enabled")
    return manager


def get_vllm_local_lora_plan(model_runner: Any, adapter_config: Mapping[str, Any]) -> Any:
    """Bind adapter targets to the actual vLLM rank's local buffers."""
    from vllm.lora.peft_helper import PEFTHelper

    manager = _get_vllm_lora_manager(model_runner)._adapter_manager
    if not hasattr(manager, "get_local_adapter_plan") or not hasattr(manager, "add_local_adapter"):
        raise RuntimeError("lora_rdt requires the compatible vLLM fork with local-adapter plan and registration APIs")
    return manager.get_local_adapter_plan(PEFTHelper.from_dict(dict(adapter_config)))


def stage_vllm_local_lora_factors(
    model_runner: Any,
    adapter_id: int,
    receiver_plan: Any,
    factors: Mapping[str, tuple[list[torch.Tensor], list[torch.Tensor]]],
) -> None:
    """Register independent local factors without activating a GPU slot."""
    manager = _get_vllm_lora_manager(model_runner)
    _validate_staging_capacity(manager, adapter_id)
    if not manager._adapter_manager.add_local_adapter(adapter_id, receiver_plan, dict(factors)):
        raise RuntimeError(f"vLLM declined local LoRA adapter id {adapter_id}")


def _validate_staging_capacity(manager: Any, adapter_id: int) -> None:
    registered = manager.list_adapters()
    if adapter_id in registered:
        raise ValueError(f"LoRA adapter id {adapter_id} is already registered")
    if len(registered) >= manager._adapter_manager.capacity:
        raise ValueError("lora_rdt requires a free registered adapter slot to retain the previous generation")
    if len(registered) >= manager._adapter_manager.lora_slots:
        raise ValueError("lora_rdt requires a free GPU adapter slot to retain the previous generation")
