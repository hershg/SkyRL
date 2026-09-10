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

__all__ = [
    "LoRAAdapterGenerationState",
    "LoRAAdapterLayout",
    "LoRATensorSlice",
    "LoRATransferInitInfo",
    "LoRAUpdateRequest",
    "build_lora_adapter_layout",
    "materialize_bf16_adapter_tensor",
]
