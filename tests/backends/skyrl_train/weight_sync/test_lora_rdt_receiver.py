from types import SimpleNamespace

import pytest
import torch

import skyrl.backends.skyrl_train.weight_sync.lora_rdt.receiver as receiver
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRAAdapterLayout,
    LoRATensorSlice,
    LoRAUpdateRequest,
    acknowledge_lora_generation,
    pull_and_stage_lora_adapter,
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
