from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    activate_staged_vllm_lora_model,
    discard_staged_vllm_lora_model,
    stage_vllm_lora_model,
)


class _AdapterManager:
    def __init__(self):
        self.adapters = {}
        self.activated = []
        self.capacity = 2
        self.lora_slots = 2

    def add_adapter(self, model):
        self.adapters[model.id] = model
        return True

    def activate_adapter(self, adapter_id):
        self.activated.append(adapter_id)
        return True


class _Manager:
    def __init__(self):
        self._adapter_manager = _AdapterManager()
        self.removed = []

    def list_adapters(self):
        return dict(self._adapter_manager.adapters)

    def remove_adapter(self, adapter_id):
        self.removed.append(adapter_id)
        self._adapter_manager.adapters.pop(adapter_id, None)


def _runner():
    return SimpleNamespace(lora_manager=_Manager())


def test_stage_registers_without_activating_and_activation_uses_the_staged_id():
    runner = _runner()
    model = SimpleNamespace(id=11)

    stage_vllm_lora_model(runner, model)

    assert runner.lora_manager.list_adapters() == {11: model}
    assert runner.lora_manager._adapter_manager.activated == []
    activate_staged_vllm_lora_model(runner, 11)
    assert runner.lora_manager._adapter_manager.activated == [11]


def test_stage_rejects_duplicate_id_without_replacing_registered_adapter():
    runner = _runner()
    original = SimpleNamespace(id=11)
    stage_vllm_lora_model(runner, original)

    with pytest.raises(ValueError, match="already registered"):
        stage_vllm_lora_model(runner, SimpleNamespace(id=11))

    assert runner.lora_manager.list_adapters() == {11: original}


def test_failed_generation_discards_only_its_staged_adapter():
    runner = _runner()
    stage_vllm_lora_model(runner, SimpleNamespace(id=11))
    stage_vllm_lora_model(runner, SimpleNamespace(id=12))

    discard_staged_vllm_lora_model(runner, 12)

    assert runner.lora_manager.list_adapters().keys() == {11}
    assert runner.lora_manager.removed == [12]


@pytest.mark.parametrize("limited_cache", ["capacity", "lora_slots"])
def test_staging_refuses_to_evict_the_previous_generation(limited_cache):
    runner = _runner()
    original = SimpleNamespace(id=11)
    stage_vllm_lora_model(runner, original)
    setattr(runner.lora_manager._adapter_manager, limited_cache, 1)

    with pytest.raises(ValueError, match="retain the previous generation"):
        stage_vllm_lora_model(runner, SimpleNamespace(id=12))

    assert runner.lora_manager.list_adapters() == {11: original}


def test_restore_accepts_an_already_active_adapter_in_real_vllm_manager():
    from vllm.lora.model_manager import AdapterLRUCache, LoRAModelManager

    adapter_manager = object.__new__(LoRAModelManager)
    adapter_manager._active_adapters = {11: None}
    adapter_manager._registered_adapters = AdapterLRUCache(2, lambda adapter_id: None)
    adapter_manager._registered_adapters[11] = SimpleNamespace(id=11)
    runner = _runner()
    runner.lora_manager._adapter_manager = adapter_manager
    runner.lora_manager.list_adapters = adapter_manager.list_adapters

    activate_staged_vllm_lora_model(runner, 11)

    assert adapter_manager._active_adapters == {11: None}


def test_staging_does_not_evict_active_adapter_from_real_vllm_cache():
    from vllm.lora.model_manager import AdapterLRUCache, LRUCacheLoRAModelManager

    adapter_manager = object.__new__(LRUCacheLoRAModelManager)
    adapter_manager.lora_config = SimpleNamespace(max_cpu_loras=1, max_loras=1)
    adapter_manager.lora_index_to_id = [11]
    adapter_manager._active_adapters = AdapterLRUCache(1, adapter_manager._deactivate_adapter)
    adapter_manager._active_adapters[11] = None
    adapter_manager._registered_adapters = AdapterLRUCache(1, adapter_manager.deactivate_adapter)
    original = SimpleNamespace(id=11)
    adapter_manager._registered_adapters[11] = original
    runner = _runner()
    runner.lora_manager._adapter_manager = adapter_manager
    runner.lora_manager.list_adapters = adapter_manager.list_adapters

    with pytest.raises(ValueError, match="retain the previous generation"):
        stage_vllm_lora_model(runner, SimpleNamespace(id=12))

    assert adapter_manager.list_adapters() == {11: original}
    assert adapter_manager.lora_index_to_id == [11]
    assert 11 in adapter_manager._active_adapters
