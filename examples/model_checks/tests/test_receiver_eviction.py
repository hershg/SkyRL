from types import SimpleNamespace

import pytest

from examples.model_checks import h1_receiver_audit as audit


@pytest.mark.parametrize("retained", [False, True])
def test_eviction_verifies_execution_slots_and_registration(monkeypatch, retained):
    manager = SimpleNamespace(lora_index_to_id=[7])
    registrations = {7}

    def remove(adapter_id):
        assert adapter_id == 7
        registrations.clear()
        if not retained:
            manager.lora_index_to_id[0] = None
        return True

    worker = object.__new__(audit.H1ReceiverAuditWorker)
    worker.model_runner = SimpleNamespace(
        get_model=lambda: SimpleNamespace(lora_manager=manager),
        remove_lora=remove,
        list_loras=lambda: registrations,
    )
    monkeypatch.setattr(audit, "get_tensor_model_parallel_rank", lambda: 3)
    if retained:
        with pytest.raises(ValueError, match="remains registered or active"):
            worker.evict_active_lora()
    else:
        assert worker.evict_active_lora() == {"adapter_id": 7, "tp_rank": 3, "evicted": True}
