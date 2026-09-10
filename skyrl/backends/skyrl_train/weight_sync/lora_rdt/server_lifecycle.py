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
        self._staged: dict[str, tuple[LoRAUpdateRequest, int, int | None]] = {}

    def get_active_adapter_id(self, adapter_name: str) -> int | None:
        """Return the currently routable adapter id for one adapter name."""
        return self._active_ids.get(adapter_name)

    async def stage(
        self,
        engine: Any,
        rendezvous: LoRardtProducerRendezvous,
        request: LoRAUpdateRequest,
        adapter_id: int,
        adapter_config: Mapping[str, Any],
    ) -> None:
        """Stage local TP buffers without changing the named request route."""
        self._validate_update(rendezvous, request, adapter_id)
        if request.adapter_name in self._staged:
            raise ValueError(f"LoRA adapter {request.adapter_name!r} already has a staged generation")
        await engine.collective_rpc(
            LORA_RDT_STAGE_METHOD,
            kwargs={
                "rendezvous": rendezvous.to_json_dict(),
                "request": request.to_json_dict(),
                "adapter_id": adapter_id,
                "adapter_config": dict(adapter_config),
            },
        )
        self._staged[request.adapter_name] = (
            request, adapter_id, self._active_ids.get(request.adapter_name)
        )

    async def activate(self, engine: Any, request: LoRAUpdateRequest, adapter_id: int) -> None:
        """Activate a staged local generation while fleet admission is closed."""
        self._get_staged(request, adapter_id)
        await engine.collective_rpc(
            LORA_RDT_ACTIVATE_METHOD,
            kwargs={"request": request.to_json_dict(), "adapter_id": adapter_id},
        )
        self._active_ids[request.adapter_name] = adapter_id

    async def commit(self, engine: Any, request: LoRAUpdateRequest, adapter_id: int) -> None:
        """Retire the prior buffer only after fleet-wide activation succeeds."""
        _, _, previous_id = self._get_staged(request, adapter_id)
        if previous_id is not None:
            await engine.collective_rpc(LORA_RDT_REMOVE_METHOD, kwargs={"adapter_id": previous_id})
        del self._staged[request.adapter_name]

    async def rollback(self, engine: Any, request: LoRAUpdateRequest, adapter_id: int) -> None:
        """Restore the prior route and discard the staged or active new buffer."""
        staged = self._staged.get(request.adapter_name)
        previous_id = staged[2] if staged is not None else None
        if previous_id is not None:
            await engine.collective_rpc(LORA_RDT_RESTORE_METHOD, kwargs={"adapter_id": previous_id})
            self._active_ids[request.adapter_name] = previous_id
        elif self._active_ids.get(request.adapter_name) == adapter_id:
            self._active_ids.pop(request.adapter_name, None)
        await engine.collective_rpc(LORA_RDT_DISCARD_METHOD, kwargs={"adapter_id": adapter_id})
        self._staged.pop(request.adapter_name, None)

    def _validate_update(self, rendezvous: LoRardtProducerRendezvous, request: LoRAUpdateRequest, adapter_id: int) -> None:
        if request.adapter_name != rendezvous.layout.adapter_name:
            raise ValueError("LoRA request adapter does not match producer rendezvous")
        if request.layout_digest != rendezvous.layout.layout_digest:
            raise ValueError("LoRA request layout digest does not match producer rendezvous")
        if request.source_dtype != rendezvous.layout.source_dtype:
            raise ValueError("LoRA request source dtype does not match producer rendezvous")
        if adapter_id <= 0:
            raise ValueError(f"LoRA adapter ids must be positive, got {adapter_id}")

    def _get_staged(self, request: LoRAUpdateRequest, adapter_id: int) -> tuple[LoRAUpdateRequest, int, int | None]:
        staged = self._staged.get(request.adapter_name)
        if staged is None or staged[:2] != (request, adapter_id):
            raise ValueError(f"LoRA adapter {request.adapter_name!r} generation {request.generation} is not staged")
        return staged

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
