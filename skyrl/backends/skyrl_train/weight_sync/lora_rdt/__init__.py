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
from .vllm_adapter import build_vllm_lora_model

__all__ = [
    "LoRAAdapterGenerationState",
    "LoRAAdapterLayout",
    "LoRATensorSlice",
    "LoRATransferInitInfo",
    "LoRAUpdateRequest",
    "build_lora_adapter_layout",
    "build_vllm_lora_model",
    "materialize_bf16_adapter_tensor",
]
