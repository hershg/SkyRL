"""Atomic named-adapter replacement over vLLM's pause and collective RPC APIs."""

import logging
from typing import Any, Mapping

from .contracts import LoRAUpdateRequest
from .control_protocol import (
    LORA_RDT_ACTIVATE_METHOD,
    LORA_RDT_DISCARD_METHOD,
    LORA_RDT_REMOVE_METHOD,
    LORA_RDT_RESTORE_METHOD,
    LORA_RDT_STAGE_METHOD,
)
from .rendezvous import LoRardtProducerRendezvous

logger = logging.getLogger(__name__)


class LoRardtRollbackError(RuntimeError):
    """The previous adapter could not be restored while requests are paused."""


class LoRardtServerLifecycle:
    """Own named active adapter ids while the server owns request admission."""

    def __init__(self) -> None:
        self._active_ids: dict[str, int] = {}

    def get_active_adapter_id(self, adapter_name: str) -> int | None:
        """Return the currently routable adapter id for one adapter name."""
        return self._active_ids.get(adapter_name)

    async def replace(
        self,
        engine: Any,
        rendezvous: LoRardtProducerRendezvous,
        request: LoRAUpdateRequest,
        adapter_id: int,
        adapter_config: Mapping[str, Any],
    ) -> int:
        """Drain, stage, activate, and retire one named adapter generation.

        ``pause_generation(mode="wait")`` closes admission and waits for existing
        requests. A collective stage then leaves the existing adapter active;
        collective activation occurs only after every worker reports stage
        success. If any later operation fails, rollback runs while admission is
        still closed. A rollback failure intentionally leaves the engine paused.
        """
        if request.adapter_name != rendezvous.layout.adapter_name:
            raise ValueError("LoRA request adapter does not match producer rendezvous")
        if request.layout_digest != rendezvous.layout.layout_digest:
            raise ValueError(
                "LoRA request layout digest does not match producer rendezvous"
            )
        if request.source_dtype != rendezvous.layout.source_dtype:
            raise ValueError(
                "LoRA request source dtype does not match producer rendezvous"
            )
        if adapter_id <= 0:
            raise ValueError(f"LoRA adapter ids must be positive, got {adapter_id}")

        adapter_name = request.adapter_name
        previous_id = self._active_ids.get(adapter_name)
        if previous_id == adapter_id:
            raise ValueError(
                f"LoRA adapter {adapter_name!r} is already active as {adapter_id}"
            )

        resume = False
        stage_attempted = False
        try:
            await engine.pause_generation(mode="wait")
            stage_attempted = True
            await engine.collective_rpc(
                LORA_RDT_STAGE_METHOD,
                kwargs={
                    "rendezvous": rendezvous.to_json_dict(),
                    "request": request.to_json_dict(),
                    "adapter_id": adapter_id,
                    "adapter_config": dict(adapter_config),
                },
            )
            await engine.collective_rpc(
                LORA_RDT_ACTIVATE_METHOD,
                kwargs={"request": request.to_json_dict(), "adapter_id": adapter_id},
            )
            if previous_id is not None:
                await engine.collective_rpc(
                    LORA_RDT_REMOVE_METHOD,
                    kwargs={"adapter_id": previous_id},
                )
            self._active_ids[adapter_name] = adapter_id
            resume = True
            return adapter_id
        except BaseException:
            try:
                if previous_id is not None:
                    await engine.collective_rpc(
                        LORA_RDT_RESTORE_METHOD,
                        kwargs={"adapter_id": previous_id},
                    )
                if stage_attempted:
                    await engine.collective_rpc(
                        LORA_RDT_DISCARD_METHOD,
                        kwargs={"adapter_id": adapter_id},
                    )
            except BaseException as rollback_error:
                logger.exception(
                    "lora_rdt rollback failed for adapter=%s generation=%s",
                    adapter_name,
                    request.generation,
                )
                raise LoRardtRollbackError(
                    f"lora_rdt rollback failed for {adapter_name!r} generation {request.generation}; "
                    "the engine remains paused"
                ) from rollback_error
            resume = True
            raise
        finally:
            if resume:
                await engine.resume_generation()
