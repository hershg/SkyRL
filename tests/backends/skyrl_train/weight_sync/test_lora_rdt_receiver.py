import logging
from types import SimpleNamespace

import pytest
import torch

import skyrl.backends.skyrl_train.weight_sync.lora_rdt.receiver as receiver
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRAAdapterLayout,
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRATensorSlice,
    LoRAUpdateRequest,
    acknowledge_lora_generation,
    pull_and_stage_lora_adapter,
    pull_reconstruct_and_stage_lora_adapter,
)


class _RemoteMethod:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return self.value


class _Producer:
    def __init__(self, tensors, acknowledgement):
        self.pull = _RemoteMethod(tensors)
        self.acknowledge = _RemoteMethod(acknowledgement)


def _layout():
    return LoRAAdapterLayout(
        adapter_name="adapter",
        source_dtype="float32",
        tensors=(
            LoRATensorSlice("a", (2,), 0, 0, 0, 0, 8),
            LoRATensorSlice("b", (2,), 1, 0, 0, 8, 8),
        ),
    )


def test_receiver_pulls_owned_slices_builds_bf16_model_and_stages(monkeypatch):
    layout = _layout()
    request = LoRAUpdateRequest.from_layout(layout, generation=2)
    producers = {
        0: _Producer({"a": torch.tensor([1.0, 2.0])}, False),
        1: _Producer({"b": torch.tensor([3.0, 4.0])}, False),
    }
    built = SimpleNamespace(id=8)
    staged = []
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)
    monkeypatch.setattr(receiver, "build_vllm_lora_model", lambda **kwargs: built)
    monkeypatch.setattr(
        receiver,
        "stage_vllm_lora_model",
        lambda runner, model: staged.append((runner, model)),
    )
    model_runner = object()

    result = pull_and_stage_lora_adapter(
        producers=producers,
        layout=layout,
        request=request,
        inference_rank=0,
        adapter_id=8,
        adapter_config={"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]},
        model_runner=model_runner,
        device="cuda",
    )

    assert result is built
    assert producers[0].pull.calls == [(2, ["a"])]
    assert producers[1].pull.calls == [(2, ["b"])]
    assert staged == [(model_runner, built)]


def test_receiver_rejects_bad_shape_before_staging(monkeypatch):
    layout = _layout()
    request = LoRAUpdateRequest.from_layout(layout, generation=2)
    producers = {
        0: _Producer({"a": torch.tensor([1.0, 2.0, 3.0])}, False),
        1: _Producer({"b": torch.tensor([3.0, 4.0])}, False),
    }
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)

    with pytest.raises(ValueError, match="shape"):
        pull_and_stage_lora_adapter(
            producers,
            layout,
            request,
            0,
            8,
            {"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]},
            object(),
            "cuda",
        )


def test_acknowledgement_waits_until_after_activation(monkeypatch):
    producers = {0: _Producer({}, False), 1: _Producer({}, True)}
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)

    assert acknowledge_lora_generation(producers, generation=2, consumer_id=7) == [
        False,
        True,
    ]
    assert producers[0].acknowledge.calls == [(2, 7)]
    assert producers[1].acknowledge.calls == [(2, 7)]


def _bridge_source(source_rank: int, tensor_parallel_rank: int) -> LoRABridgeSource:
    return LoRABridgeSource(
        key="decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight",
        source_rank=source_rank,
        hf_param_names=("base_model.model.layers.0.mlp.down_proj.lora_B.weight",),
        component="linear_out",
        transform="identity",
        shape=(1, 2),
        tensor_parallel_axis=0,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=2,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=(),
    )


def test_receiver_pulls_bridge_sources_reconstructs_and_stages(monkeypatch):
    layout = LoRABridgeSourceLayout(
        "adapter", (_bridge_source(0, 0), _bridge_source(1, 1))
    )
    request = LoRAUpdateRequest(
        adapter_name="adapter",
        generation=2,
        layout_digest=layout.layout_digest,
        source_dtype="float32",
    )
    source_key = layout.sources[0].key
    producers = {
        0: _Producer({source_key: torch.tensor([[1.0, 2.0]])}, False),
        1: _Producer({source_key: torch.tensor([[3.0, 4.0]])}, False),
    }
    captured = {}
    built = SimpleNamespace(id=8)
    staged = []
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        receiver,
        "build_vllm_lora_model",
        lambda **kwargs: captured.update(kwargs) or built,
    )
    monkeypatch.setattr(
        receiver,
        "stage_vllm_lora_model",
        lambda runner, model: staged.append((runner, model)),
    )
    model_runner = object()

    result = pull_reconstruct_and_stage_lora_adapter(
        producers=producers,
        layout=layout,
        request=request,
        adapter_id=8,
        adapter_config={"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]},
        model_runner=model_runner,
        device="cuda",
    )

    assert result is built
    assert producers[0].pull.calls == [(2, [source_key])]
    assert producers[1].pull.calls == [(2, [source_key])]
    assert torch.equal(
        captured["source_tensors"][
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight"
        ],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert (
        captured["source_tensors"][
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight"
        ].dtype
        is torch.float32
    )
    assert captured["dtype"] is torch.bfloat16
    assert staged == [(model_runner, built)]


def test_receiver_rejects_bridge_source_layout_mismatch_without_pulling(monkeypatch):
    layout = LoRABridgeSourceLayout(
        "adapter", (_bridge_source(0, 0), _bridge_source(1, 1))
    )
    request = LoRAUpdateRequest(
        adapter_name="adapter",
        generation=2,
        layout_digest="0" * 64,
        source_dtype="float32",
    )
    producer = _Producer({}, False)
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)

    with pytest.raises(ValueError, match="digest"):
        pull_reconstruct_and_stage_lora_adapter(
            producers={0: producer},
            layout=layout,
            request=request,
            adapter_id=8,
            adapter_config={"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]},
            model_runner=object(),
            device="cuda",
        )

    assert producer.pull.calls == []


def test_local_receiver_emits_reconciled_stage_receipts(monkeypatch, caplog):
    from skyrl.backends.skyrl_train.weight_sync.lora_rdt import consumer_plan
    from skyrl.backends.skyrl_train.weight_sync.lora_rdt.consumer_plan import (
        LoRAConsumerPlan,
        LoRAConsumerPull,
    )
    from skyrl.backends.skyrl_train.weight_sync.lora_rdt.contracts import (
        LoRASourceSlice,
    )

    source = _bridge_source(0, 0)
    source = LoRABridgeSource(
        key=source.key,
        source_rank=0,
        hf_param_names=source.hf_param_names,
        component=source.component,
        transform=source.transform,
        shape=(1, 2),
        tensor_parallel_axis=None,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=source.transform_config,
    )
    layout = LoRABridgeSourceLayout("adapter", (source,))
    selection = LoRASourceSlice(source.key, (0, 0), (1, 2))
    plan = LoRAConsumerPlan(
        layout.layout_digest,
        receiver_plan=object(),
        pulls=(LoRAConsumerPull(0, selection),),
        copies=(),
    )
    producer = SimpleNamespace(
        pull_slices=_RemoteMethod([torch.ones((1, 2), dtype=torch.float32)])
    )
    staged = []
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)
    monkeypatch.setattr(
        consumer_plan,
        "assemble_lora_consumer_factors",
        lambda plan, pulled, device: {"factor": ([], [])},
    )
    monkeypatch.setattr(
        receiver,
        "stage_vllm_local_lora_factors",
        lambda runner, adapter_id, receiver_plan, factors: staged.append(adapter_id),
    )
    caplog.set_level(
        logging.INFO,
        logger="skyrl.backends.skyrl_train.weight_sync.lora_rdt.receiver",
    )

    receiver.pull_and_stage_local_lora_adapter(
        {0: producer},
        layout,
        LoRAUpdateRequest.from_layout(layout, 3),
        plan,
        9,
        object(),
        "cpu",
    )

    receipts = [
        record.message
        for record in caplog.records
        if "lora_rdt_receiver_stage" in record.message
    ]
    assert any(
        "generation=3" in message
        and "phase=nixl_pull" in message
        and "source_bytes=8" in message
        for message in receipts
    )
    assert any("phase=bf16_assembly_submit" in message for message in receipts)
    assert any("phase=vllm_registration_submit" in message for message in receipts)
    assert any("phase=local_stage_envelope" in message for message in receipts)
    assert staged == [9]
