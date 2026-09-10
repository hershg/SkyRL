"""Trainer-side fixed-layout publication planning for LoRA RDT."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import ray
import torch

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
    """Build immutable generations from one rank-local Bridge export.

    Distributed collectives and Ray actor creation stay outside this class. That
    lets Megatron own process-group ordering while this planner enforces the
    fixed-layout and monotonically increasing-generation contracts.
    """

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
        self._layout_digest: str | None = None
        self._generation = -1

    def plan(
        self,
        records: Iterable[Any],
        gathered_sources: Sequence[Sequence[LoRABridgeSource]],
        gathered_actor_names: Sequence[str],
    ) -> LoRardtPublication:
        """Validate a rank-local export and build its next fleet-visible generation."""
        local_tensors, local_sources = extract_lora_bridge_sources(
            records, self._source_rank
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
        if self._layout_digest is None:
            self._layout_digest = layout.layout_digest
        elif self._layout_digest != layout.layout_digest:
            raise ValueError(
                f"lora_rdt adapter {self._adapter_name!r} changed its fixed source layout"
            )
        if len(gathered_actor_names) != len(
            {source.source_rank for source in layout.sources}
        ):
            raise ValueError(
                "lora_rdt requires exactly one producer actor name per source rank"
            )
        self._generation += 1
        request = LoRAUpdateRequest.from_layout(layout, self._generation)
        rendezvous = LoRardtProducerRendezvous(
            layout=layout,
            producer_actor_names=tuple(enumerate(gathered_actor_names)),
            consumer_count=self._consumer_count,
            namespace=self._namespace,
        )
        return LoRardtPublication(local_tensors, layout, request, rendezvous)


def publish_lora_sources(
    producer: Any,
    publication: LoRardtPublication,
    timeout_seconds: float,
) -> None:
    """Require every trainer rank to retain its sources before receivers start."""
    error = None
    try:
        ray.get(
            producer.publish.remote(
                publication.request,
                publication.local_tensors,
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
