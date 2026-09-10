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
            raise ValueError(
                f"lora_rdt requires float32 source tensor {name!r}, got {tensor.dtype}"
            )
    peft_helper = PEFTHelper.from_dict(dict(adapter_config))
    return LoRAModel.from_lora_tensors(
        adapter_id,
        {
            name: tensor.to(device=device, dtype=dtype).clone()
            for name, tensor in source_tensors.items()
        },
        peft_helper,
        device=device,
        dtype=dtype,
        model_vocab_size=model_vocab_size,
        weights_mapper=weights_mapper,
        skip_prefixes=skip_prefixes,
    )
