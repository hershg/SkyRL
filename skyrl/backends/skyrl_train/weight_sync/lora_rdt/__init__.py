"""Contracts for the LoRA-specific NIXL/RDMA weight-sync backend."""

from .contracts import (
    LoRAAdapterGenerationState,
    LoRAAdapterLayout,
    LoRATensorSlice,
    LoRATransferInitInfo,
    LoRAUpdateRequest,
    build_lora_adapter_layout,
    materialize_bf16_adapter_tensor,
)
from .producer import LoRardtProducer
from .vllm_adapter import (
    activate_staged_vllm_lora_model,
    build_vllm_lora_model,
    discard_staged_vllm_lora_model,
    stage_vllm_lora_model,
)

__all__ = [
    "LoRAAdapterGenerationState",
    "LoRAAdapterLayout",
    "LoRATensorSlice",
    "LoRardtProducer",
    "LoRATransferInitInfo",
    "LoRAUpdateRequest",
    "activate_staged_vllm_lora_model",
    "build_lora_adapter_layout",
    "build_vllm_lora_model",
    "discard_staged_vllm_lora_model",
    "materialize_bf16_adapter_tensor",
    "stage_vllm_lora_model",
]
