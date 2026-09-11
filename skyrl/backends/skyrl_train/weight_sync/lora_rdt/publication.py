"""Trainer-side fixed-layout publication planning for LoRA RDT."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from .bridge_sources import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    extract_lora_bridge_sources,
)
from .contracts import LoRAUpdateRequest
from .rendezvous import LoRardtProducerRendezvous


@dataclass(frozen=True)
class LoRardtPublication:
    """One rank's source snapshot plus fleet-visible rendezvous metadata."""

    local_tensors: dict[str, torch.Tensor]
    layout: LoRABridgeSourceLayout
    request: LoRAUpdateRequest
    rendezvous: LoRardtProducerRendezvous


def make_lora_rdt_producer_name(
    layout: LoRABridgeSourceLayout, source_rank: int
) -> str:
    """Return a deterministic actor name for one fixed layout and trainer rank."""
    if source_rank < 0:
        raise ValueError(
            f"lora_rdt source rank must be non-negative, got {source_rank}"
        )
    return f"skyrl_lora_rdt_{layout.layout_digest[:16]}_rk{source_rank}"


class LoRardtPublicationPlanner:
    """Build immutable generations for one fixed global Bridge layout."""

    def __init__(
        self,
        adapter_name: str,
        source_rank: int,
        consumer_count: int,
        namespace: str | None,
    ) -> None:
        if consumer_count <= 0:
            raise ValueError("lora_rdt requires at least one inference deployment")
        self._adapter_name = adapter_name
        self._source_rank = source_rank
        self._consumer_count = consumer_count
        self._namespace = namespace
        self._layout: LoRABridgeSourceLayout | None = None
        self._rendezvous: LoRardtProducerRendezvous | None = None
        self._generation = -1

    def plan(
        self,
        records: Iterable[Any],
        gathered_sources: Sequence[Sequence[LoRABridgeSource]] | None = None,
        gathered_actor_names: Sequence[str] | None = None,
    ) -> LoRardtPublication:
        """Validate fresh local values and build the next generation."""
        local_tensors, local_sources = extract_lora_bridge_sources(
            records, self._source_rank
        )
        if self._layout is None:
            self._initialize(gathered_sources, gathered_actor_names)
        elif gathered_sources is not None or gathered_actor_names is not None:
            raise ValueError(
                "lora_rdt static publication metadata is already initialized"
            )

        assert self._layout is not None
        assert self._rendezvous is not None
        expected_local_sources = tuple(
            source
            for source in self._layout.sources
            if source.source_rank == self._source_rank
        )
        if local_sources != expected_local_sources:
            raise ValueError(
                f"lora_rdt adapter {self._adapter_name!r} changed its fixed local source layout"
            )

        self._generation += 1
        request = LoRAUpdateRequest.from_layout(self._layout, self._generation)
        return LoRardtPublication(
            local_tensors, self._layout, request, self._rendezvous
        )

    def _initialize(
        self,
        gathered_sources: Sequence[Sequence[LoRABridgeSource]] | None,
        gathered_actor_names: Sequence[str] | None,
    ) -> None:
        """Freeze global layout and producer rendezvous on first publication."""
        if gathered_sources is None or gathered_actor_names is None:
            raise ValueError(
                "lora_rdt first publication requires global sources and producer names"
            )
        sources = tuple(
            sorted(
                (
                    source
                    for rank_sources in gathered_sources
                    for source in rank_sources
                ),
                key=lambda source: (
                    source.key,
                    source.expert_parallel_rank,
                    source.tensor_parallel_rank,
                    source.source_rank,
                ),
            )
        )
        layout = LoRABridgeSourceLayout(self._adapter_name, sources)
        if len(gathered_actor_names) != len(
            {source.source_rank for source in layout.sources}
        ):
            raise ValueError(
                "lora_rdt requires exactly one producer actor name per source rank"
            )
        self._layout = layout
        self._rendezvous = LoRardtProducerRendezvous(
            layout=layout,
            producer_actor_names=tuple(enumerate(gathered_actor_names)),
            consumer_count=self._consumer_count,
            namespace=self._namespace,
        )


def publish_lora_sources(
    producer: Any,
    publication: LoRardtPublication,
    timeout_seconds: float,
) -> None:
    """Require every trainer rank to retain its sources before receivers start."""
    error = None
    try:
        handles = export_lora_cuda_ipc(publication.local_tensors)
        ray.get(
            producer.publish_cuda.remote(
                publication.request,
                handles,
                publication.rendezvous.consumer_count,
            ),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    request = publication.request
    receipt = (request.generation, request.layout_digest, error)
    receipts = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(receipts, receipt)
    failures = [
        f"rank {rank}: {result}"
        for rank, result in enumerate(receipts)
        if result != (request.generation, request.layout_digest, None)
    ]
    if not failures:
        return
    cleanup_error = None
    try:
        ray.get(producer.discard.remote(request.generation), timeout=timeout_seconds)
    except Exception as exc:
        cleanup_error = f"{type(exc).__name__}: {exc}"
    cleanup_errors = [None] * len(receipts)
    torch.distributed.all_gather_object(cleanup_errors, cleanup_error)
    failures.extend(
        f"rank {rank} cleanup: {result}"
        for rank, result in enumerate(cleanup_errors)
        if result is not None
    )
    raise RuntimeError("LoRA RDT sources not ready: " + "; ".join(failures))


def export_lora_cuda_ipc(tensors: dict[str, torch.Tensor]) -> dict[str, tuple]:
    """Share retained trainer snapshots with a sidecar on the same physical GPU."""
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.dtype is not torch.float32:
            raise ValueError(f"LoRA IPC source {name!r} must be a CUDA float32 tensor")
    for device in {tensor.device for tensor in tensors.values()}:
        torch.cuda.current_stream(device).synchronize()
    return {name: reduce_tensor(tensor.detach())[1] for name, tensor in tensors.items()}
