import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.lora.local_adapter", reason="requires the pinned local-adapter vLLM fork")

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import ColumnParallelLinearWithLoRA
from vllm.lora.model_manager import AdapterLRUCache, LoRAModelManager

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRAUpdateRequest,
    receiver,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.consumer_plan import (
    build_lora_consumer_plan,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.vllm_adapter import (
    activate_staged_vllm_lora_model,
    discard_staged_vllm_lora_model,
    get_vllm_local_lora_plan,
)


class _SliceProducer:
    def __init__(self, tensors):
        self.tensors = tensors
        self.calls = []
        self.failure = None
        self.pull_slices = SimpleNamespace(remote=self._pull_slices)

    def _pull_slices(self, generation, selections):
        self.calls.append((generation, selections))
        values = [self.tensors[item.key][item.indices].clone() for item in selections]
        if self.failure == "count":
            return values[:-1]
        if self.failure == "dtype":
            values[-1] = values[-1].to(torch.bfloat16)
        if self.failure == "shape":
            values[-1] = values[-1].reshape(-1)
        return values


def _create_receiver():
    config = LoRAConfig(max_lora_rank=8, max_loras=2, max_cpu_loras=2, lora_dtype=torch.bfloat16)
    layer = object.__new__(ColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.lora_config = config
    layer.base_layer = SimpleNamespace(input_size=8, output_size=8)
    layer.tp_rank, layer.tp_size = 1, 2
    layer.n_slices = 1
    layer.is_merged_col_linear = False
    layer.lora_a_stacked = (torch.zeros(2, 1, 8, 8, dtype=torch.bfloat16),)
    layer.lora_b_stacked = (torch.zeros(2, 1, 4, 8, dtype=torch.bfloat16),)
    manager = object.__new__(LoRAModelManager)
    manager.lora_config = config
    manager.modules = {"model.proj": layer}
    manager.packed_modules = {}
    manager.lora_index_to_id = [None, None]
    manager._active_adapters = {}
    manager._registered_adapters = AdapterLRUCache(2, manager.deactivate_adapter)
    worker_manager = SimpleNamespace(
        _adapter_manager=manager, list_adapters=manager.list_adapters, remove_adapter=manager.remove_adapter
    )
    return SimpleNamespace(lora_manager=worker_manager), manager, layer


def _create_sources():
    full = {
        "A": torch.arange(32, dtype=torch.float32).reshape(4, 8) / 17,
        "B": torch.arange(32, 64, dtype=torch.float32).reshape(8, 4) / 19,
    }
    sources, producers = [], {}
    for rank in range(2):
        tensors = {}
        for component, axis in (("A", 0), ("B", 1)):
            value = full[component].chunk(2, dim=axis)[rank].clone()
            tensors[component] = value
            sources.append(
                LoRABridgeSource(
                    component,
                    rank,
                    (f"model.proj.lora_{component}.weight",),
                    "linear_in" if component == "A" else "linear_out",
                    "identity",
                    tuple(value.shape),
                    axis,
                    rank,
                    2,
                    None,
                    0,
                    1,
                    (),
                )
            )
        producers[rank] = _SliceProducer(tensors)
    sources.sort(key=lambda item: (item.key, item.expert_parallel_rank, item.tensor_parallel_rank, item.source_rank))
    return LoRABridgeSourceLayout("adapter", tuple(sources)), producers, full


def _prepare(monkeypatch):
    runner, manager, layer = _create_receiver()
    layout, producers, full = _create_sources()
    config = {"r": 4, "lora_alpha": 8, "target_modules": ["proj"]}
    plan = build_lora_consumer_plan(layout, get_vllm_local_lora_plan(runner, config))
    monkeypatch.setattr(receiver.ray, "get", lambda values: values)
    return runner, manager, layer, layout, producers, full, plan


def test_sliced_receiver_registers_and_activates_exact_local_values_once(monkeypatch):
    runner, manager, layer, layout, producers, full, plan = _prepare(monkeypatch)
    request = LoRAUpdateRequest.from_layout(layout, 1)
    receiver.pull_and_stage_local_lora_adapter(producers, layout, request, plan, 11, runner, "cpu")
    assert manager.lora_index_to_id == [None, None]
    assert torch.count_nonzero(layer.lora_a_stacked[0]) == 0
    assert plan.source_bytes == 192
    assert sum(item.numel() * 4 for producer in producers.values() for item in producer.tensors.values()) == 256
    assert all(len(producer.calls) == 1 for producer in producers.values())
    activate_staged_vllm_lora_model(runner, 11)
    activate_staged_vllm_lora_model(runner, 11)
    torch.testing.assert_close(layer.lora_a_stacked[0][0, 0, :4], full["A"].bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(layer.lora_b_stacked[0][0, 0, :, :4], full["B"][4:].bfloat16() * 2, rtol=0, atol=0)
    assert manager.lora_index_to_id == [11, None]
    for producer in producers.values():
        for tensor in producer.tensors.values():
            tensor.zero_()
    torch.testing.assert_close(layer.lora_a_stacked[0][0, 0, :4], full["A"].bfloat16(), rtol=0, atol=0)
    discard_staged_vllm_lora_model(runner, 11)
    assert manager.list_adapters() == {}
    assert manager.lora_index_to_id == [None, None]


@pytest.mark.parametrize("failure", ["count", "dtype", "shape", "digest", "missing_producer"])
def test_invalid_slice_transfer_keeps_the_previous_active_adapter(monkeypatch, failure):
    runner, manager, layer, layout, producers, full, plan = _prepare(monkeypatch)
    receiver.pull_and_stage_local_lora_adapter(
        producers, layout, LoRAUpdateRequest.from_layout(layout, 1), plan, 11, runner, "cpu"
    )
    activate_staged_vllm_lora_model(runner, 11)
    before = (layer.lora_a_stacked[0].clone(), layer.lora_b_stacked[0].clone())
    if failure == "digest":
        plan = replace(plan, source_layout_digest="0" * 64)
    elif failure == "missing_producer":
        del producers[1]
    else:
        producers[1].failure = failure
    with pytest.raises(ValueError):
        receiver.pull_and_stage_local_lora_adapter(
            producers, layout, LoRAUpdateRequest.from_layout(layout, 2), plan, 12, runner, "cpu"
        )
    assert set(manager.list_adapters()) == {11}
    assert manager.lora_index_to_id == [11, None]
    torch.testing.assert_close(layer.lora_a_stacked[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(layer.lora_b_stacked[0], before[1], rtol=0, atol=0)


def test_receiver_holds_transport_and_staging_buffers_until_cuda_completion(monkeypatch):
    from skyrl.backends.skyrl_train.weight_sync.lora_rdt import consumer_plan

    runner, manager, layer, layout, producers, full, plan = _prepare(monkeypatch)
    buffer_refs = []
    original_assemble = consumer_plan.assemble_lora_consumer_factors
    fences = []

    def assemble_on_cpu(plan, pulled, device):
        factors = original_assemble(plan, pulled, "cpu")
        buffer_refs.extend(weakref.ref(value) for value in pulled.values())
        buffer_refs.extend(weakref.ref(value) for pair in factors.values() for component in pair for value in component)
        return factors

    def complete_copies(device):
        assert device == "cuda:0"
        assert buffer_refs and all(ref() is not None for ref in buffer_refs)
        assert manager.lora_index_to_id == [None, None]
        fences.append(device)

    monkeypatch.setattr(consumer_plan, "assemble_lora_consumer_factors", assemble_on_cpu)
    monkeypatch.setattr(receiver.torch.cuda, "synchronize", complete_copies)
    receiver.pull_and_stage_local_lora_adapter(
        producers, layout, LoRAUpdateRequest.from_layout(layout, 1), plan, 11, runner, "cuda:0"
    )
    assert fences == ["cuda:0"]
    assert all(ref() is None for ref in buffer_refs)
    assert set(manager.list_adapters()) == {11}
