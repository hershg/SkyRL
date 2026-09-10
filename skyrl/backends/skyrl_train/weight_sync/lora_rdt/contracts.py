"""Immutable layout and generation contracts for LoRA RDT publication."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class LoRATensorSlice:
    """One source-to-destination tensor slice in the fixed adapter layout."""

    key: str
    shape: tuple[int, ...]
    source_rank: int
    source_offset: int
    destination_rank: int
    destination_offset: int
    byte_length: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("LoRA tensor slices require a non-empty key")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError(
                f"LoRA tensor {self.key!r} has an invalid shape {self.shape!r}"
            )
        if (
            min(
                self.source_rank,
                self.source_offset,
                self.destination_rank,
                self.destination_offset,
            )
            < 0
        ):
            raise ValueError(f"LoRA tensor {self.key!r} has a negative rank or offset")
        if self.byte_length <= 0:
            raise ValueError(f"LoRA tensor {self.key!r} has a non-positive byte length")


@dataclass(frozen=True)
class LoRAAdapterLayout:
    """The fixed, content-independent transport layout for one named adapter."""

    adapter_name: str
    source_dtype: str
    tensors: tuple[LoRATensorSlice, ...]
    layout_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ValueError("LoRA adapter layouts require a non-empty adapter name")
        if self.source_dtype != "float32":
            raise ValueError(
                f"lora_rdt requires float32 sources, got {self.source_dtype!r}"
            )
        if not self.tensors:
            raise ValueError("LoRA adapter layouts require at least one tensor")
        keys = [tensor.key for tensor in self.tensors]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError(
                "LoRA adapter layout tensor keys must be sorted and unique"
            )
        for tensor in self.tensors:
            expected_bytes = (
                prod(tensor.shape)
                * torch.tensor([], dtype=torch.float32).element_size()
            )
            if tensor.byte_length != expected_bytes:
                raise ValueError(
                    f"LoRA tensor {tensor.key!r} has byte_length={tensor.byte_length}, expected {expected_bytes} "
                    "for a float32 source"
                )
        payload = {
            "adapter_name": self.adapter_name,
            "source_dtype": self.source_dtype,
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        object.__setattr__(self, "layout_digest", hashlib.sha256(encoded).hexdigest())


def build_lora_adapter_layout(
    adapter_name: str,
    tensors: Mapping[str, torch.Tensor],
    ownership: Mapping[str, tuple[int, int]],
) -> LoRAAdapterLayout:
    """Build the canonical layout without materializing or inspecting tensor values."""
    if set(tensors) != set(ownership):
        missing_ownership = sorted(set(tensors) - set(ownership))
        missing_tensors = sorted(set(ownership) - set(tensors))
        raise ValueError(
            f"LoRA layout ownership must match tensors; missing ownership={missing_ownership}, "
            f"missing tensors={missing_tensors}"
        )
    source_offsets: dict[int, int] = {}
    destination_offsets: dict[int, int] = {}
    slices = []
    for key in sorted(tensors):
        tensor = tensors[key]
        if tensor.dtype is not torch.float32:
            raise ValueError(
                f"lora_rdt requires float32 source tensor {key!r}, got {tensor.dtype}"
            )
        source_rank, destination_rank = ownership[key]
        byte_length = tensor.numel() * tensor.element_size()
        source_offset = source_offsets.get(source_rank, 0)
        destination_offset = destination_offsets.get(destination_rank, 0)
        slices.append(
            LoRATensorSlice(
                key=key,
                shape=tuple(tensor.shape),
                source_rank=source_rank,
                source_offset=source_offset,
                destination_rank=destination_rank,
                destination_offset=destination_offset,
                byte_length=byte_length,
            )
        )
        source_offsets[source_rank] = source_offset + byte_length
        destination_offsets[destination_rank] = destination_offset + byte_length
    return LoRAAdapterLayout(
        adapter_name=adapter_name, source_dtype="float32", tensors=tuple(slices)
    )


@dataclass(frozen=True)
class LoRATransferInitInfo:
    """Static information shared while initializing a LoRA RDT receiver."""

    layout: LoRAAdapterLayout
    inference_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.inference_ranks:
            raise ValueError("lora_rdt requires at least one inference rank")
        if tuple(sorted(self.inference_ranks)) != self.inference_ranks:
            raise ValueError("lora_rdt inference ranks must be sorted")
        if len(set(self.inference_ranks)) != len(self.inference_ranks):
            raise ValueError("lora_rdt inference ranks must be unique")


@dataclass(frozen=True)
class LoRAUpdateRequest:
    """A single named-adapter publication generation."""

    adapter_name: str
    generation: int
    layout_digest: str
    source_dtype: str

    @classmethod
    def from_layout(
        cls, layout: LoRAAdapterLayout, generation: int
    ) -> "LoRAUpdateRequest":
        return cls(
            adapter_name=layout.adapter_name,
            generation=generation,
            layout_digest=layout.layout_digest,
            source_dtype=layout.source_dtype,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "LoRAUpdateRequest":
        return cls(**data)

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ValueError("LoRA update requests require a non-empty adapter name")
        if self.generation < 0:
            raise ValueError(
                f"LoRA generation must be non-negative, got {self.generation}"
            )
        if self.source_dtype != "float32":
            raise ValueError(
                f"lora_rdt requires float32 sources, got {self.source_dtype!r}"
            )
        if len(self.layout_digest) != 64:
            raise ValueError("LoRA update requests require a SHA-256 layout digest")


@dataclass(frozen=True)
class LoRAReceiverGeneration:
    """Retain the fixed receiver contract with its active or staged generation."""

    request: LoRAUpdateRequest
    adapter_id: int
    adapter_config_json: str


class LoRAAdapterGenerationState:
    """Tracks staged and active adapter buffers for a fixed receiver layout."""

    def __init__(self, init_info: LoRATransferInitInfo) -> None:
        self._init_info = init_info
        self._active_generation: int | None = None
        self._active_buffers: dict[int, Any] = {}
        self._staged_buffers: dict[int, dict[int, Any]] = {}

    @property
    def active_generation(self) -> int | None:
        return self._active_generation

    @property
    def active_buffers(self) -> Mapping[int, Any]:
        return dict(self._active_buffers)

    def stage(
        self, request: LoRAUpdateRequest, inference_rank: int, buffer: Any
    ) -> None:
        """Record one fully validated staging buffer before group activation."""
        self._validate_request(request)
        if inference_rank not in self._init_info.inference_ranks:
            raise ValueError(
                f"Inference rank {inference_rank} is not part of this adapter receiver"
            )
        if buffer is None:
            raise ValueError("lora_rdt cannot stage an empty adapter buffer")
        if (
            self._active_generation is not None
            and request.generation <= self._active_generation
        ):
            raise ValueError(
                f"LoRA generation {request.generation} is stale; active generation is {self._active_generation}"
            )
        if self._staged_buffers and request.generation not in self._staged_buffers:
            staged_generation = next(iter(self._staged_buffers))
            raise ValueError(
                f"LoRA generation {staged_generation} is still staging; cannot stage generation {request.generation}"
            )
        self._staged_buffers.setdefault(request.generation, {})[inference_rank] = buffer

    def activate(self, request: LoRAUpdateRequest) -> Mapping[int, Any]:
        """Make one fully acknowledged generation active at a request boundary."""
        self._validate_request(request)
        staged = self._staged_buffers.get(request.generation)
        expected_ranks = set(self._init_info.inference_ranks)
        if staged is None or set(staged) != expected_ranks:
            acknowledged = set() if staged is None else set(staged)
            missing = sorted(expected_ranks - acknowledged)
            raise ValueError(
                f"LoRA generation {request.generation} cannot activate before acknowledgements from ranks {missing}"
            )
        self._active_generation = request.generation
        self._active_buffers = staged
        self._staged_buffers = {}
        return self.active_buffers

    def discard(self, request: LoRAUpdateRequest) -> None:
        """Discard a failed staging generation without disturbing active buffers."""
        self._validate_request(request)
        self._staged_buffers.pop(request.generation, None)

    def unload(self) -> None:
        """Release active and staged state after callers have drained requests."""
        self._active_generation = None
        self._active_buffers = {}
        self._staged_buffers = {}

    def _validate_request(self, request: LoRAUpdateRequest) -> None:
        layout = self._init_info.layout
        if request.adapter_name != layout.adapter_name:
            raise ValueError(
                f"LoRA request adapter {request.adapter_name!r} does not match {layout.adapter_name!r}"
            )
        if request.layout_digest != layout.layout_digest:
            raise ValueError(
                "LoRA request layout digest does not match the initialized receiver layout"
            )
        if request.source_dtype != layout.source_dtype:
            raise ValueError(
                "LoRA request source dtype does not match the initialized receiver layout"
            )


def materialize_bf16_adapter_tensor(source: torch.Tensor) -> torch.Tensor:
    """Create independent BF16 inference storage from one FP32 source tensor."""
    if source.dtype is not torch.float32:
        raise ValueError(f"lora_rdt requires float32 sources, got {source.dtype}")
    return source.to(dtype=torch.bfloat16).clone()
