"""Serializable named-producer rendezvous for one fixed LoRA RDT layout."""

from dataclasses import dataclass
from typing import Any, Mapping

from .bridge_sources import LoRABridgeSourceLayout


@dataclass(frozen=True)
class LoRardtProducerRendezvous:
    """The immutable control-plane information a receiver needs to find sources."""

    layout: LoRABridgeSourceLayout
    producer_actor_names: tuple[tuple[int, str], ...]
    consumer_count: int
    namespace: str | None = None

    def __post_init__(self) -> None:
        if self.consumer_count <= 0:
            raise ValueError("lora_rdt requires at least one inference consumer")
        ranks = tuple(rank for rank, _ in self.producer_actor_names)
        expected_ranks = tuple(
            sorted({source.source_rank for source in self.layout.sources})
        )
        if ranks != expected_ranks:
            raise ValueError(
                f"LoRA RDT producer ranks must be {expected_ranks}, got {ranks}"
            )
        names = tuple(name for _, name in self.producer_actor_names)
        if not all(names) or len(names) != len(set(names)):
            raise ValueError(
                "LoRA RDT producer actor names must be non-empty and unique"
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-safe rendezvous metadata for an inference control RPC."""
        return {
            "layout": self.layout.to_json_dict(),
            "producer_actor_names": list(self.producer_actor_names),
            "consumer_count": self.consumer_count,
            "namespace": self.namespace,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "LoRardtProducerRendezvous":
        """Rebuild and verify a rendezvous received by an inference worker."""
        return cls(
            layout=LoRABridgeSourceLayout.from_json_dict(data["layout"]),
            producer_actor_names=tuple(
                (int(rank), str(name)) for rank, name in data["producer_actor_names"]
            ),
            consumer_count=int(data["consumer_count"]),
            namespace=data["namespace"],
        )

    def actor_name_by_rank(self) -> dict[int, str]:
        """Return the validated source-rank to Ray actor-name mapping."""
        return dict(self.producer_actor_names)
