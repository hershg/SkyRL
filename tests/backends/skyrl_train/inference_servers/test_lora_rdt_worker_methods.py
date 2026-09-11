import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap as worker_wrap
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRardtProducerRendezvous,
    LoRAUpdateRequest,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.contracts import (
    LoRAReceiverGeneration,
)


@pytest.fixture(autouse=True)
def _mock_cuda_fence(monkeypatch):
    monkeypatch.setattr(worker_wrap.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.build_vllm_lora_consumer_plan",
        lambda layout, config, runner: SimpleNamespace(source_layout_digest=layout.layout_digest),
    )


def _record(generation, adapter_id, config=None):
    config = {"r": 2} if config is None else config
    return LoRAReceiverGeneration(
        _request(generation), adapter_id, json.dumps(config, sort_keys=True, separators=(",", ":"))
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
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_and_stage_local_lora_adapter",
        lambda **kwargs: staged.append(kwargs),
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.activate_staged_vllm_lora_model",
        lambda runner, adapter_id: activated.append((runner, adapter_id)),
    )
    request = _request(2)

    result = worker.stage_lora_rdt_adapter(_rendezvous().to_json_dict(), request.to_json_dict(), 9, {"r": 2})

    assert result == {"adapter_id": 9, "generation": 2}
    assert staged[0]["device"] == "cuda:0"
    assert worker._skyrl_lora_rdt_staged == {"adapter": _record(2, 9)}

    worker.activate_lora_rdt_adapter(request.to_json_dict(), 9)

    assert activated == [(worker.model_runner, 9)]
    assert worker._skyrl_lora_rdt_active == {"adapter": _record(2, 9)}
    assert worker._skyrl_lora_rdt_staged == {}


def test_worker_rejects_stale_or_unstaged_generation(monkeypatch):
    worker = _worker()
    worker._skyrl_lora_rdt_active = {"adapter": _record(2, 9)}
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        lambda names, namespace: {0: "producer"},
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_and_stage_local_lora_adapter",
        lambda **kwargs: None,
    )

    with pytest.raises(ValueError, match="stale"):
        worker.stage_lora_rdt_adapter(_rendezvous().to_json_dict(), _request(1).to_json_dict(), 10, {"r": 2})
    with pytest.raises(ValueError, match="not staged"):
        worker.activate_lora_rdt_adapter(_request(3).to_json_dict(), 10)


def test_worker_rollback_restores_generation_and_allows_a_replacement(monkeypatch):
    worker = _worker()
    registered = set()
    plan_builds = []

    def build_plan(layout, config, runner):
        plan = SimpleNamespace(source_layout_digest=layout.layout_digest)
        plan_builds.append(plan)
        return plan

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.build_vllm_lora_consumer_plan",
        build_plan,
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        lambda names, namespace: {0: "producer"},
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_and_stage_local_lora_adapter",
        lambda **kwargs: registered.add(kwargs["adapter_id"]),
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.activate_staged_vllm_lora_model",
        lambda runner, adapter_id: None,
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.discard_staged_vllm_lora_model",
        lambda runner, adapter_id: registered.discard(adapter_id),
    )
    for generation, adapter_id in ((1, 9), (2, 10)):
        worker.stage_lora_rdt_adapter(
            _rendezvous().to_json_dict(), _request(generation).to_json_dict(), adapter_id, {"r": 2}
        )
        worker.activate_lora_rdt_adapter(_request(generation).to_json_dict(), adapter_id)

    worker.restore_lora_rdt_adapter(9)
    worker.discard_lora_rdt_adapter(10)
    worker.discard_lora_rdt_adapter(10)

    assert worker._skyrl_lora_rdt_active == {"adapter": _record(1, 9)}
    assert registered == {9}
    assert worker._skyrl_lora_rdt_retained == {}
    worker.stage_lora_rdt_adapter(_rendezvous().to_json_dict(), _request(2).to_json_dict(), 11, {"r": 2})
    worker.activate_lora_rdt_adapter(_request(2).to_json_dict(), 11)
    worker.remove_lora_rdt_adapter(9)

    assert worker._skyrl_lora_rdt_active == {"adapter": _record(2, 11)}
    assert worker._skyrl_lora_rdt_retained == {}
    assert registered == {11}

    worker.remove_lora_rdt_adapter(11)

    assert worker._skyrl_lora_rdt_active == {}
    assert registered == set()
    assert worker._skyrl_lora_rdt_consumer_plans == {}
    assert len(plan_builds) == 1


def test_worker_restoring_unknown_generation_does_not_touch_vllm(monkeypatch):
    worker = _worker()

    def unexpected_activation(runner, adapter_id):
        raise AssertionError("an unknown generation must not activate")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.activate_staged_vllm_lora_model",
        unexpected_activation,
    )
    with pytest.raises(ValueError, match="no retained generation"):
        worker.restore_lora_rdt_adapter(9)


@pytest.mark.parametrize("change", ["layout", "rank", "targets"])
def test_changed_receiver_contract_is_rejected_before_any_pull(monkeypatch, change):
    worker = _worker()
    config = {"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]}
    original = _record(1, 9, config)
    worker._skyrl_lora_rdt_active = {"adapter": original}
    rendezvous = _rendezvous()
    if change == "layout":
        source = replace(rendezvous.layout.sources[0], shape=(2, 2))
        layout = LoRABridgeSourceLayout("adapter", (source,))
        rendezvous = LoRardtProducerRendezvous(layout, ((0, "producer-0"),), 1)
    elif change == "rank":
        config["r"] = 4
    else:
        config["target_modules"] = ["q_proj"]

    def unexpected_resolve(*args):
        raise AssertionError("changed contracts must be rejected before producer lookup")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        unexpected_resolve,
    )

    with pytest.raises(ValueError, match="changed its fixed receiver"):
        worker.stage_lora_rdt_adapter(
            rendezvous.to_json_dict(),
            LoRAUpdateRequest.from_layout(rendezvous.layout, 2).to_json_dict(),
            10,
            config,
        )

    assert worker._skyrl_lora_rdt_active == {"adapter": original}
    assert not getattr(worker, "_skyrl_lora_rdt_staged", {})


def test_activation_rejects_request_with_a_different_staged_digest(monkeypatch):
    worker = _worker()
    worker._skyrl_lora_rdt_staged = {"adapter": _record(2, 10)}
    worker._skyrl_lora_rdt_active = {"adapter": _record(1, 9)}

    def unexpected_activation(*args):
        raise AssertionError("mismatched staged request must not activate")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.activate_staged_vllm_lora_model",
        unexpected_activation,
    )
    mismatched = replace(_request(2), layout_digest="0" * 64)
    with pytest.raises(ValueError, match="not staged"):
        worker.activate_lora_rdt_adapter(mismatched.to_json_dict(), 10)

    assert worker._skyrl_lora_rdt_active == {"adapter": _record(1, 9)}
    assert worker._skyrl_lora_rdt_staged == {"adapter": _record(2, 10)}


@pytest.mark.parametrize("phase", ["stage", "activate", "restore"])
def test_failed_cuda_completion_does_not_acknowledge_generation(monkeypatch, phase):
    worker = _worker()
    old_record = _record(1, 9)
    next_record = _record(2, 10)
    worker._skyrl_lora_rdt_active = {"adapter": old_record}
    worker._skyrl_lora_rdt_staged = {}
    worker._skyrl_lora_rdt_retained = {}
    queued = []
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        lambda names, namespace: {0: "producer"},
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_and_stage_local_lora_adapter",
        lambda **kwargs: queued.append(kwargs["adapter_id"]),
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.activate_staged_vllm_lora_model",
        lambda runner, adapter_id: queued.append(adapter_id),
    )
    if phase == "activate":
        worker._skyrl_lora_rdt_staged["adapter"] = next_record
    elif phase == "restore":
        worker._skyrl_lora_rdt_active["adapter"] = next_record
        worker._skyrl_lora_rdt_retained[9] = ("adapter", old_record)
    active_before = worker._skyrl_lora_rdt_active.copy()
    staged_before = worker._skyrl_lora_rdt_staged.copy()
    retained_before = worker._skyrl_lora_rdt_retained.copy()

    def fail_completion(device):
        assert device == worker.device
        assert queued == [9 if phase == "restore" else 10]
        assert worker._skyrl_lora_rdt_active == active_before
        assert worker._skyrl_lora_rdt_staged == staged_before
        raise RuntimeError("injected asynchronous copy failure")

    monkeypatch.setattr(worker_wrap.torch.cuda, "synchronize", fail_completion)
    with pytest.raises(RuntimeError, match="asynchronous copy failure"):
        if phase == "stage":
            worker.stage_lora_rdt_adapter(_rendezvous().to_json_dict(), _request(2).to_json_dict(), 10, {"r": 2})
        elif phase == "activate":
            worker.activate_lora_rdt_adapter(_request(2).to_json_dict(), 10)
        else:
            worker.restore_lora_rdt_adapter(9)

    assert worker._skyrl_lora_rdt_active == active_before
    assert worker._skyrl_lora_rdt_staged == staged_before
    assert worker._skyrl_lora_rdt_retained == retained_before


@pytest.mark.parametrize("field", ["adapter_name", "layout_digest"])
def test_worker_rejects_request_mismatch_before_plan_or_actor_lookup(monkeypatch, field):
    worker = _worker()
    request = replace(_request(1), **{field: "wrong-name" if field == "adapter_name" else "0" * 64})

    def unexpected_plan(*args):
        raise AssertionError("invalid requests must not build a plan")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.build_vllm_lora_consumer_plan",
        unexpected_plan,
    )
    with pytest.raises(ValueError, match="does not match"):
        worker.stage_lora_rdt_adapter(_rendezvous().to_json_dict(), request.to_json_dict(), 9, {"r": 2})
    assert not getattr(worker, "_skyrl_lora_rdt_consumer_plans", {})


@pytest.mark.parametrize("failure_phase", ["pull", "completion"])
def test_failed_first_stage_does_not_retain_a_plan(monkeypatch, failure_phase):
    worker = _worker()
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.resolve_lora_rdt_producers",
        lambda names, namespace: {0: "producer"},
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.pull_and_stage_local_lora_adapter",
        fail if failure_phase == "pull" else lambda **kwargs: None,
    )
    if failure_phase == "completion":
        monkeypatch.setattr(worker_wrap.torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        worker.stage_lora_rdt_adapter(_rendezvous().to_json_dict(), _request(1).to_json_dict(), 9, {"r": 2})
    assert not getattr(worker, "_skyrl_lora_rdt_consumer_plans", {})
    assert not getattr(worker, "_skyrl_lora_rdt_staged", {})


def test_discarding_the_only_staged_generation_releases_its_plan(monkeypatch):
    worker = _worker()
    worker._skyrl_lora_rdt_staged = {"adapter": _record(1, 9)}
    worker._skyrl_lora_rdt_consumer_plans = {"adapter": object()}
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_rdt.discard_staged_vllm_lora_model",
        lambda runner, adapter_id: None,
    )
    worker.discard_lora_rdt_adapter(9)
    assert worker._skyrl_lora_rdt_consumer_plans == {}
    assert worker._skyrl_lora_rdt_staged == {}
