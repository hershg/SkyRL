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
