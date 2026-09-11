"""Atomic named-adapter replacement over vLLM's pause and collective RPC APIs."""

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


class LoRardtRollbackError(RuntimeError):
    """A generation could not be safely restored or discarded."""


class LoRardtServerLifecycle:
    """Own named active adapter ids while the server owns request admission."""

    def __init__(self) -> None:
        self._active_ids: dict[str, int] = {}
        self._staged: dict[str, tuple[LoRAUpdateRequest, int, int | None]] = {}
        self._terminal: dict[str, tuple[LoRAUpdateRequest, int, str]] = {}
        self._unloaded: set[str] = set()

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
        if request.adapter_name in self._unloaded:
            raise ValueError(f"LoRA adapter {request.adapter_name!r} has been unloaded")
        self._validate_update(rendezvous, request, adapter_id)
        if self._get_terminal_outcome(request, adapter_id) is not None:
            raise ValueError(f"LoRA generation {request.generation} has already completed")
        if request.adapter_name in self._staged:
            raise ValueError(f"LoRA adapter {request.adapter_name!r} already has a staged generation")
        try:
            await engine.collective_rpc(
                LORA_RDT_STAGE_METHOD,
                kwargs={
                    "rendezvous": rendezvous.to_json_dict(),
                    "request": request.to_json_dict(),
                    "adapter_id": adapter_id,
                    "adapter_config": dict(adapter_config),
                },
            )
        except BaseException:
            try:
                await engine.collective_rpc(LORA_RDT_DISCARD_METHOD, kwargs={"adapter_id": adapter_id})
            except BaseException as cleanup_error:
                raise LoRardtRollbackError(
                    f"lora_rdt could not discard partially staged adapter id {adapter_id}"
                ) from cleanup_error
            raise
        self._staged[request.adapter_name] = (request, adapter_id, self._active_ids.get(request.adapter_name))

    async def activate(self, engine: Any, request: LoRAUpdateRequest, adapter_id: int) -> None:
        """Activate a staged local generation while fleet admission is closed."""
        self._get_staged(request, adapter_id)
        if self._active_ids.get(request.adapter_name) == adapter_id:
            return
        await engine.collective_rpc(
            LORA_RDT_ACTIVATE_METHOD,
            kwargs={"request": request.to_json_dict(), "adapter_id": adapter_id},
        )
        self._active_ids[request.adapter_name] = adapter_id

    async def commit(self, engine: Any, request: LoRAUpdateRequest, adapter_id: int) -> bool:
        """Retire the prior buffer once and report whether the route needs finalizing."""
        outcome = self._get_terminal_outcome(request, adapter_id)
        if outcome == "committed":
            return False
        if outcome is not None:
            raise ValueError(f"LoRA generation {request.generation} was already rolled back")
        _, _, previous_id = self._get_staged(request, adapter_id)
        if self._active_ids.get(request.adapter_name) != adapter_id:
            raise ValueError(f"LoRA generation {request.generation} has not been activated")
        if previous_id is not None:
            await engine.collective_rpc(LORA_RDT_REMOVE_METHOD, kwargs={"adapter_id": previous_id})
        del self._staged[request.adapter_name]
        self._terminal[request.adapter_name] = (request, adapter_id, "committed")
        return True

    async def rollback(self, engine: Any, request: LoRAUpdateRequest, adapter_id: int) -> bool:
        """Restore the prior route once without replaying destructive cleanup."""
        outcome = self._get_terminal_outcome(request, adapter_id)
        if outcome == "rolled_back":
            return False
        if outcome is not None:
            raise ValueError(f"LoRA generation {request.generation} was already committed")
        _, _, previous_id = self._get_staged(request, adapter_id)
        if previous_id is not None:
            await engine.collective_rpc(LORA_RDT_RESTORE_METHOD, kwargs={"adapter_id": previous_id})
            self._active_ids[request.adapter_name] = previous_id
        elif self._active_ids.get(request.adapter_name) == adapter_id:
            self._active_ids.pop(request.adapter_name, None)
        await engine.collective_rpc(LORA_RDT_DISCARD_METHOD, kwargs={"adapter_id": adapter_id})
        self._staged.pop(request.adapter_name, None)
        self._terminal[request.adapter_name] = (request, adapter_id, "rolled_back")
        return True

    async def unload(self, engine: Any, adapter_name: str) -> None:
        """Release the active buffer while admission is closed, retaining a name tombstone."""
        if adapter_name in self._staged:
            raise ValueError(f"LoRA adapter {adapter_name!r} has an unfinished replacement")
        self._unloaded.add(adapter_name)
        adapter_id = self._active_ids.get(adapter_name)
        if adapter_id is not None:
            await engine.collective_rpc(LORA_RDT_REMOVE_METHOD, kwargs={"adapter_id": adapter_id})
        self._active_ids.pop(adapter_name, None)
        self._terminal.pop(adapter_name, None)

    def _get_terminal_outcome(self, request: LoRAUpdateRequest, adapter_id: int) -> str | None:
        terminal = self._terminal.get(request.adapter_name)
        if terminal is None:
            return None
        previous_request, previous_id, outcome = terminal
        if request.generation < previous_request.generation:
            raise ValueError(f"LoRA generation {request.generation} is older than the last completed transaction")
        if request.generation == previous_request.generation:
            if request != previous_request or adapter_id != previous_id:
                raise ValueError(f"LoRA generation {request.generation} does not match its completed transaction")
            return outcome
        return None

    def _validate_update(
        self, rendezvous: LoRardtProducerRendezvous, request: LoRAUpdateRequest, adapter_id: int
    ) -> None:
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
