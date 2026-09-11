"""Rank-local NIXL producer for one fixed LoRA adapter layout."""

from dataclasses import dataclass, field
from threading import Lock
from typing import Mapping

import ray
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from .bridge_sources import LoRABridgeSourceLayout
from .contracts import LoRAAdapterLayout, LoRASourceSlice, LoRAUpdateRequest


@dataclass
class _PublishedGeneration:
    tensors: dict[str, torch.Tensor]
    acknowledgements: set[int]
    consumer_count: int
    slices: dict[LoRASourceSlice, torch.Tensor] = field(default_factory=dict)


class LoRardtProducer:
    """Retain one rank's FP32 LoRA tensors until every receiver acknowledges them."""

    def __init__(
        self,
        source_rank: int,
        layout: LoRAAdapterLayout | LoRABridgeSourceLayout,
    ) -> None:
        self._lock = Lock()
        self._source_rank = source_rank
        self._layout = layout
        self._generations: dict[int, _PublishedGeneration] = {}
        self._latest_generation: int | None = None

    def publish(
        self,
        request: LoRAUpdateRequest,
        tensors: Mapping[str, torch.Tensor],
        consumer_count: int,
    ) -> None:
        with self._lock:
            self._validate_request(request)
            if consumer_count <= 0:
                raise ValueError(f"lora_rdt requires a positive consumer count, got {consumer_count}")
            if request.generation in self._generations:
                raise ValueError(f"LoRA generation {request.generation} is already published")
            if self._latest_generation is not None and request.generation <= self._latest_generation:
                raise ValueError(
                    f"LoRA generation {request.generation} is stale; latest generation is {self._latest_generation}"
                )
            expected = self._owned_tensor_shapes()
            if set(tensors) != set(expected):
                raise ValueError(
                    f"LoRA producer rank {self._source_rank} expected tensors {sorted(expected)}, got {sorted(tensors)}"
                )
            for name, tensor in tensors.items():
                if tensor.dtype is not torch.float32:
                    raise ValueError(f"lora_rdt requires float32 source tensor {name!r}, got {tensor.dtype}")
                if tuple(tensor.shape) != expected[name]:
                    raise ValueError(
                        f"LoRA producer rank {self._source_rank} returned {name!r} with shape "
                        f"{tuple(tensor.shape)}, expected {expected[name]}"
                    )
            self._generations[request.generation] = _PublishedGeneration(dict(tensors), set(), consumer_count)
            self._latest_generation = request.generation

    def publish_cuda(
        self,
        request: LoRAUpdateRequest,
        handles: Mapping[str, tuple],
        consumer_count: int,
    ) -> None:
        """Rebuild trainer-local CUDA IPC handles before exposing tensors over NIXL."""
        tensors = {}
        for name, arguments in handles.items():
            arguments = list(arguments)
            # The sidecar sees the trainer's physical GPU as local device zero.
            arguments[6] = torch.cuda.current_device()
            tensors[name] = rebuild_cuda_tensor(*arguments)
        # Consume IPC references even when a timeout has already discarded the generation.
        self.publish(request, tensors, consumer_count)

    @ray.method(tensor_transport="nixl")
    def pull(self, generation: int, names: list[str]) -> dict[str, torch.Tensor]:
        """Return requested rank-owned source tensors through Ray NIXL transport."""
        with self._lock:
            published = self._generations.get(generation)
            if published is None:
                raise ValueError(f"LoRA generation {generation} is not retained by producer rank {self._source_rank}")
            unknown = set(names) - set(published.tensors)
            if unknown:
                raise ValueError(f"LoRA producer rank {self._source_rank} does not own tensors {sorted(unknown)}")
            return {name: published.tensors[name] for name in names}

    @ray.method(tensor_transport="nixl")
    def pull_slices(self, generation: int, slices: list[LoRASourceSlice]) -> list[torch.Tensor]:
        """Pack only requested rectangles into retained, exact-sized FP32 allocations."""
        with self._lock:
            published = self._generations.get(generation)
            if published is None:
                raise ValueError(f"LoRA generation {generation} is not retained by producer rank {self._source_rank}")
            if not slices or len(set(slices)) != len(slices):
                raise ValueError("LoRA slice pulls require unique nonempty selections")
            for selection in slices:
                if selection.key not in published.tensors:
                    raise ValueError(f"LoRA producer rank {self._source_rank} does not own {selection.key!r}")
                selection.validate_shape(tuple(published.tensors[selection.key].shape))
            copied_devices = set()
            for selection in slices:
                if selection not in published.slices:
                    source = published.tensors[selection.key]
                    # contiguous() alone can retain a view's full source allocation.
                    packed = source[selection.indices].clone(memory_format=torch.contiguous_format)
                    published.slices[selection] = packed
                    if packed.is_cuda:
                        copied_devices.add(packed.device)
            for device in copied_devices:
                torch.cuda.current_stream(device).synchronize()
            return [published.slices[selection] for selection in slices]

    def acknowledge(self, generation: int, consumer_id: int) -> bool:
        """Record one completed receiver and release source storage after the final acknowledgement."""
        with self._lock:
            published = self._generations.get(generation)
            if published is None:
                raise ValueError(f"LoRA generation {generation} is not retained by producer rank {self._source_rank}")
            if consumer_id in published.acknowledgements:
                return False
            published.acknowledgements.add(consumer_id)
            if len(published.acknowledgements) == published.consumer_count:
                del self._generations[generation]
                return True
            return False

    def discard(self, generation: int) -> None:
        """Release an unpublished or failed generation without touching another generation."""
        with self._lock:
            self._generations.pop(generation, None)
            if self._latest_generation is None or generation > self._latest_generation:
                self._latest_generation = generation

    def retained_generations(self) -> list[int]:
        """Return generations whose source buffers remain available to receivers."""
        with self._lock:
            return sorted(self._generations)

    def _owned_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return this producer's fixed source names and shapes."""
        if isinstance(self._layout, LoRABridgeSourceLayout):
            owned = [source for source in self._layout.sources if source.source_rank == self._source_rank]
            return {source.key: source.shape for source in owned}
        return {tensor.key: tensor.shape for tensor in self._layout.tensors if tensor.source_rank == self._source_rank}

    def _validate_request(self, request: LoRAUpdateRequest) -> None:
        if request.adapter_name != self._layout.adapter_name:
            raise ValueError(
                f"LoRA request adapter {request.adapter_name!r} does not match {self._layout.adapter_name!r}"
            )
        if request.layout_digest != self._layout.layout_digest:
            raise ValueError("LoRA request layout digest does not match producer layout")
        if request.source_dtype != self._layout.source_dtype:
            raise ValueError("LoRA request source dtype does not match producer layout")
