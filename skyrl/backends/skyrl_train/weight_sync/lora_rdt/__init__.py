"""Contracts for the LoRA-specific NIXL/RDMA weight-sync backend."""

from .bridge_sources import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    extract_lora_bridge_sources,
    reconstruct_lora_bridge_tensors,
    validate_lora_bridge_source_layout,
)
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
from .receiver import (
    acknowledge_lora_generation,
    pull_and_stage_lora_adapter,
    pull_reconstruct_and_stage_lora_adapter,
    resolve_lora_rdt_producers,
)
from .rendezvous import LoRardtProducerRendezvous
from .vllm_adapter import (
    activate_staged_vllm_lora_model,
    build_vllm_lora_model,
    discard_staged_vllm_lora_model,
    stage_vllm_lora_model,
)

__all__ = [
    "reconstruct_lora_bridge_tensors",
    "validate_lora_bridge_source_layout",
    "extract_lora_bridge_sources",
    "LoRABridgeSource",
    "LoRABridgeSourceLayout",
    "LoRAAdapterGenerationState",
    "LoRAAdapterLayout",
    "LoRATensorSlice",
    "LoRardtProducer",
    "LoRardtProducerRendezvous",
    "LoRATransferInitInfo",
    "LoRAUpdateRequest",
    "acknowledge_lora_generation",
    "activate_staged_vllm_lora_model",
    "build_lora_adapter_layout",
    "build_vllm_lora_model",
    "discard_staged_vllm_lora_model",
    "materialize_bf16_adapter_tensor",
    "pull_and_stage_lora_adapter",
    "pull_reconstruct_and_stage_lora_adapter",
    "resolve_lora_rdt_producers",
    "stage_vllm_lora_model",
]
