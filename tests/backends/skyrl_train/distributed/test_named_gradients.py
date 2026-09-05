import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron import named_gradients
from skyrl.backends.skyrl_train.distributed.megatron.named_gradients import (
    NamedGradientRecorder,
)
from skyrl.train.config import (
    MTPConfig,
    OptimizerConfig,
    PolicyConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)


class _Chunk(torch.nn.Module):
    def __init__(self, **parameters):
        super().__init__()
        for name, value in parameters.items():
            self.register_parameter(name, torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)))


class _Optimizer:
    def __init__(self, parameters, group="model", events=None):
        self._parameters = list(parameters)
        self._group = group
        self._events = events if events is not None else []
        self.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)

    def get_parameters(self):
        return self._parameters

    def _filter_grads_for_norm(self, parameters):
        return [param.grad for param in parameters if param.grad is not None and not getattr(param, "duplicate", False)]

    def get_grad_stats_parallel_group(self):
        return self._group

    def get_grad_norm(self):
        self._events.append("global_norm")
        return 1.0

    def prepare_grads(self):
        self._events.append("prepare")
        return False

    def step(self):
        self.prepare_grads()
        value = self.get_grad_norm()
        self._events.extend(("clip", "mutate"))
        return value


def _cpu_norm(grads, grad_stats_parallel_group):
    del grad_stats_parallel_group
    return math.sqrt(sum(float(torch.sum(grad.float() ** 2)) for grad in grads))


def test_named_gradient_selection_metric_names_and_order(monkeypatch):
    chunk = _Chunk(first=[3.0], second=[4.0])
    for param in chunk.parameters():
        param.grad = param.detach().clone()
    events = []
    optimizer = _Optimizer(chunk.parameters(), events=events)
    recorder = NamedGradientRecorder(
        optimizer,
        [chunk],
        {"first_weight": "first", "second_weight": "second"},
    )
    monkeypatch.setattr(named_gradients, "get_grad_norm_fp32", _cpu_norm)

    with recorder.capture(optimizer) as metrics:
        optimizer.step()

    assert events == ["prepare", "global_norm", "clip", "mutate"]
    assert metrics == {
        "skyrl.ai/named_grad_norm/first_weight": 3.0,
        "skyrl.ai/named_grad_norm/second_weight": 4.0,
    }


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_named_gradient_zero_and_nonzero(monkeypatch, value):
    chunk = _Chunk(weight=[value])
    chunk.weight.grad = chunk.weight.detach().clone()
    optimizer = _Optimizer(chunk.parameters())
    recorder = NamedGradientRecorder(optimizer, [chunk], {"weight": "weight"})
    monkeypatch.setattr(named_gradients, "get_grad_norm_fp32", _cpu_norm)

    with recorder.capture(optimizer) as metrics:
        optimizer.step()

    assert metrics["skyrl.ai/named_grad_norm/weight"] == value


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_named_gradient_nonfinite_fails_before_clip_and_mutation(monkeypatch, value):
    chunk = _Chunk(weight=[value])
    chunk.weight.grad = chunk.weight.detach().clone()
    events = []
    optimizer = _Optimizer(chunk.parameters(), events=events)
    recorder = NamedGradientRecorder(optimizer, [chunk], {"weight": "weight"})
    monkeypatch.setattr(named_gradients, "get_grad_norm_fp32", _cpu_norm)

    with pytest.raises(FloatingPointError, match="non-finite"):
        with recorder.capture(optimizer):
            optimizer.step()

    assert events == ["prepare"]


def test_named_gradient_combines_optimizer_shards_without_duplicates(monkeypatch):
    first = _Chunk(weight=[3.0], duplicate=[12.0])
    second = _Chunk(weight=[4.0])
    first.duplicate.duplicate = True
    for param in (*first.parameters(), *second.parameters()):
        param.grad = param.detach().clone()
    dense = _Optimizer(first.parameters(), group="tp_pp")
    expert = _Optimizer(second.parameters(), group="tp_ep_pp")
    optimizer = SimpleNamespace(
        chained_optimizers=[dense, expert],
        prepare_grads=lambda: False,
        get_grad_norm=lambda: 13.0,
    )
    recorder = NamedGradientRecorder(optimizer, [first, second], {"all_weights": "*"})
    calls = []

    def norm(grads, grad_stats_parallel_group):
        calls.append(grad_stats_parallel_group)
        return _cpu_norm(grads, grad_stats_parallel_group)

    monkeypatch.setattr(named_gradients, "get_grad_norm_fp32", norm)
    with recorder.capture(optimizer) as metrics:
        optimizer.prepare_grads()

    assert calls == ["tp_pp", "tp_ep_pp"]
    assert metrics["skyrl.ai/named_grad_norm/all_weights"] == 5.0


def test_named_gradient_uses_only_locally_owned_optimizer_shards(monkeypatch):
    chunk = _Chunk(first=[3.0], owned_on_other_dp_rank=[4.0])
    shard = torch.nn.Parameter(torch.tensor([0.0]))
    shard.grad = torch.tensor([3.0])
    optimizer = _Optimizer([shard])
    optimizer.model_float16_groups = [[chunk.first]]
    optimizer.model_fp32_groups = [[]]
    optimizer.shard_fp32_from_float16_groups = [[shard]]
    optimizer.shard_fp32_groups = [[]]
    recorder = NamedGradientRecorder(optimizer, [chunk], {"weights": "*"})
    monkeypatch.setattr(named_gradients, "get_grad_norm_fp32", _cpu_norm)

    with recorder.capture(optimizer) as metrics:
        optimizer.step()

    assert metrics["skyrl.ai/named_grad_norm/weights"] == 3.0


def test_named_gradient_selector_validation():
    assert OptimizerConfig().named_gradient_selectors == {}
    with pytest.raises(ValueError, match="at most 8"):
        OptimizerConfig(named_gradient_selectors={str(i): "*" for i in range(9)})
    with pytest.raises(ValueError, match="stable name"):
        OptimizerConfig(named_gradient_selectors={"bad/name": "*"})
    with pytest.raises(ValueError, match="stable name"):
        OptimizerConfig(named_gradient_selectors={"µ": "*"})
    with pytest.raises(ValueError, match="non-empty"):
        OptimizerConfig(named_gradient_selectors={"weight": ""})

    policy = PolicyConfig(optimizer_config=OptimizerConfig(named_gradient_selectors={"weight": "*"}))
    with pytest.raises(ValueError, match="requires trainer.strategy=megatron"):
        SkyRLTrainConfig(trainer=TrainerConfig(policy=policy))
    with pytest.raises(ValueError, match="does not yet support trainer.mtp.enabled"):
        SkyRLTrainConfig(
            trainer=TrainerConfig(strategy="megatron", policy=policy, mtp=MTPConfig(enabled=True))
        )

    chunk = _Chunk(weight=[1.0])
    optimizer = _Optimizer(chunk.parameters())
    with pytest.raises(ValueError, match="matched no parameters"):
        NamedGradientRecorder(optimizer, [chunk], {"missing": "bias"})
    with pytest.raises(ValueError, match="must not overlap"):
        NamedGradientRecorder(optimizer, [chunk], {"one": "weight", "two": "*"})


def test_optimizer_metrics_propagate_from_worker_to_client(monkeypatch):
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronPolicyWorkerBase,
    )
    from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
    from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend
    from skyrl.tinker import types

    chunk = _Chunk(weight=[3.0])
    chunk.weight.grad = chunk.weight.detach().clone()
    chunk.zero_grad_buffer = MagicMock()
    events = []
    optimizer = _Optimizer(chunk.parameters(), events=events)
    worker = MegatronPolicyWorkerBase.__new__(MegatronPolicyWorkerBase)
    worker.optimizer = optimizer
    worker.model = SimpleNamespace(run_pending_grad_sync=MagicMock(side_effect=lambda: events.append("sync")))
    worker.strategy = SimpleNamespace(
        optimizer_step=lambda optimizer, *_args, **_kwargs: torch.tensor(optimizer.step())
    )
    worker.scheduler = object()
    worker.actor_module = [chunk]
    worker._micro_batches_accumulated = 1
    worker._named_gradient_recorder = NamedGradientRecorder(
        optimizer,
        [chunk],
        {"mlp": "weight"},
    )
    monkeypatch.setattr(named_gradients, "get_grad_norm_fp32", _cpu_norm)
    worker_metrics = worker.optim_step(return_metrics=True)
    assert worker_metrics == {
        "skyrl.ai/grad_norm": 1.0,
        "skyrl.ai/named_grad_norm/mlp": 3.0,
    }
    worker.model.run_pending_grad_sync.assert_called_once_with()
    chunk.zero_grad_buffer.assert_called_once_with()
    assert events == ["sync", "prepare", "global_norm", "clip", "mutate"]

    actor_group = SimpleNamespace(
        async_run_ray_method=MagicMock(return_value=[worker_metrics]),
    )
    dispatch = WorkerDispatch.__new__(WorkerDispatch)
    dispatch._actor_groups = {"policy": actor_group}
    dispatch._ensure_on_gpu = MagicMock()
    dispatch.ensure_active_adapter = MagicMock()
    dispatch._save_memory_snapshot = MagicMock()
    monkeypatch.setattr("skyrl.backends.skyrl_train.workers.worker_dispatch.ray.get", lambda value: value)

    backend = SkyRLTrainBackend.__new__(SkyRLTrainBackend)
    backend._dispatch = dispatch
    backend._cfg = SimpleNamespace(trainer=SimpleNamespace(strategy="megatron"))
    backend._get_role = MagicMock(return_value="policy")
    output = backend.optim_step(
        "model",
        types.OptimStepInput(
            adam_params=types.AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                weight_decay=0.0,
            )
        ),
    )
    client_output = types.OptimStepOutput.model_validate_json(output.model_dump_json())

    assert client_output.metrics == {
        **worker_metrics,
        "skyrl.ai/learning_rate": 1e-4,
    }
    actor_group.async_run_ray_method.assert_any_call("pass_through", "optim_step", True)
