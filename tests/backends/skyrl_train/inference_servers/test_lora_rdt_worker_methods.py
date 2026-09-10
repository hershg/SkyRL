import pytest

import skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap as worker_wrap
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRardtProducerRendezvous,
    LoRAUpdateRequest,
)


def _rendezvous():
    layout = LoRABridgeSourceLayout(
        "adapter",
        (
            LoRABridgeSource(
                key="adapter.weight",
                source_rank=0,
                hf_param_names=("adapter.weight",),
                component="linear_out",
                transform="identity",
                shape=(1, 2),
                tensor_parallel_axis=0,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                expert_parallel_axis=None,
                expert_parallel_rank=0,
                expert_parallel_size=1,
                transform_config=(),
            ),
        ),
    )
    return LoRardtProducerRendezvous(layout, ((0, "producer-0"),), 1)


def _worker():
    worker = object.__new__(worker_wrap.NewInferenceWorkerWrap)
    worker.model_runner = object()
    worker.device = "cuda:0"
    return worker


def _request(generation):
    return LoRAUpdateRequest.from_layout(
        _rendezvous().layout,
        generation=generation,
    )


def test_worker_stages_then_activates_only_the_requested_generation(monkeypatch):
    worker = _worker()
    staged = []
    activated = []
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        lambda names, namespace: {0: "producer"},
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_reconstruct_and_stage_lora_adapter",
        lambda **kwargs: staged.append(kwargs),
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.activate_staged_vllm_lora_model",
        lambda runner, adapter_id: activated.append((runner, adapter_id)),
    )
    request = _request(2)

    result = worker.stage_lora_rdt_adapter(
        _rendezvous().to_json_dict(), request.to_json_dict(), 9, {"r": 2}
    )

    assert result == {"adapter_id": 9, "generation": 2}
    assert staged[0]["device"] == "cuda:0"
    assert worker._skyrl_lora_rdt_staged == {"adapter": (2, 9)}

    worker.activate_lora_rdt_adapter(request.to_json_dict(), 9)

    assert activated == [(worker.model_runner, 9)]
    assert worker._skyrl_lora_rdt_active == {"adapter": (2, 9)}
    assert worker._skyrl_lora_rdt_staged == {}


def test_worker_rejects_stale_or_unstaged_generation(monkeypatch):
    worker = _worker()
    worker._skyrl_lora_rdt_active = {"adapter": (2, 9)}
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        lambda names, namespace: {0: "producer"},
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_reconstruct_and_stage_lora_adapter",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="stale"):
        worker.stage_lora_rdt_adapter(
            _rendezvous().to_json_dict(), _request(1).to_json_dict(), 10, {"r": 2}
        )
    with pytest.raises(ValueError, match="not staged"):
        worker.activate_lora_rdt_adapter(_request(3).to_json_dict(), 10)
