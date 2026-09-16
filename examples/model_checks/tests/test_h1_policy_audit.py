"""CPU checks for H1 policy audits."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ray")
pytest.importorskip("megatron.core")
pytest.importorskip("vllm")

from examples.model_checks.h1_policy_audit import (
    TARGET_MODULES,
    H1PolicyAuditWorker,
    classify_targets,
    collect_optimizer_state,
    fingerprint_state,
    get_adapter_parameters,
    iter_state_tensors,
)


def test_target_inventory_requires_every_glm_adapter_category():
    names = [f"decoder.layers.0.self_attention.{target}.adapter.linear_in.weight" for target in TARGET_MODULES[:5]]
    names += [f"decoder.layers.0.mlp.{target}.adapter.linear_in.weight" for target in TARGET_MODULES[-2:]]
    names += [f"decoder.layers.1.mlp.experts.{target}.adapter.linear_in.weight" for target in TARGET_MODULES[-2:]]
    names += [
        f"decoder.layers.1.mlp.shared_experts.{target}.adapter.linear_in.weight" for target in TARGET_MODULES[-2:]
    ]
    coverage = classify_targets(names)
    assert all(coverage["targets"].values())
    assert all(coverage["categories"].values())


def test_frozen_adapter_parameters_remain_visible_to_audit():
    adapter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    frozen_adapter = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    base = torch.nn.Parameter(torch.ones(2))
    chunk = SimpleNamespace(
        named_parameters=lambda: [
            ("linear_q_down_proj.adapter.linear_in.weight", adapter),
            ("linear_q_down_proj.adapter.linear_out.weight", frozen_adapter),
            ("linear_q_down_proj.weight", base),
        ]
    )
    assert get_adapter_parameters([chunk]) == [
        ("chunk0.linear_q_down_proj.adapter.linear_in.weight", adapter),
        ("chunk0.linear_q_down_proj.adapter.linear_out.weight", frozen_adapter),
    ]


def test_state_tensor_walk_preserves_nested_names():
    state = {"state": [{"exp_avg": torch.ones(2)}], "step": 1}
    assert [name for name, _ in iter_state_tensors(state)] == ["state.0.exp_avg"]


def build_distributed_optimizer():
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range

    parameter = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    adam = torch.optim.Adam([parameter], lr=1e-6)
    parameter.sum().backward()
    adam.step()
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.optimizer = adam
    optimizer.is_stub_optimizer = False
    optimizer.grad_scaler = None
    optimizer.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
    dtype = (torch.float32, torch.float32)
    optimizer.per_bucket_numel = [{dtype: [2]}]
    optimizer.per_bucket_numel_unpadded = [{dtype: [2]}]
    optimizer.gbuf_ranges = [{dtype: [{"param_map": {parameter: {"gbuf_local": Range(0, 2)}}}]}]
    optimizer.model_param_group_index_map = {parameter: (0, 0)}
    return optimizer, parameter


def test_optimizer_audit_reads_real_distributed_moments(monkeypatch):
    optimizer, parameter = build_distributed_optimizer()
    worker = object.__new__(H1PolicyAuditWorker)
    worker.optimizer = optimizer
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    assert not list(iter_state_tensors(optimizer.state_dict()))
    assert worker.describe_optimizer_state()["passed"]
    assert worker.describe_optimizer_state()["moment_count"] == 2
    before = fingerprint_state(collect_optimizer_state(optimizer))
    optimizer.optimizer.state[parameter]["exp_avg"][0] += 1
    assert fingerprint_state(collect_optimizer_state(optimizer)) != before
    optimizer.optimizer.state[parameter]["exp_avg_sq"][0] = torch.nan
    assert not worker.describe_optimizer_state()["passed"]


@pytest.mark.parametrize("missing", ["exp_avg", "exp_avg_sq"])
def test_missing_distributed_moment_fails_closed(monkeypatch, missing):
    optimizer, parameter = build_distributed_optimizer()
    worker = object.__new__(H1PolicyAuditWorker)
    worker.optimizer = optimizer
    del optimizer.optimizer.state[parameter][missing]
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    with pytest.raises(KeyError, match=missing):
        worker.describe_optimizer_state()


def test_checkpoint_probe_mutates_actual_adam_moments_and_scheduler(monkeypatch):
    from copy import deepcopy

    from megatron.core.optimizer.optimizer import ChainedOptimizer

    from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
        get_megatron_optimizer_param_scheduler,
    )

    optimizer, parameter = build_distributed_optimizer()
    worker = object.__new__(H1PolicyAuditWorker)
    worker.optimizer = ChainedOptimizer([optimizer])
    worker.scheduler = get_megatron_optimizer_param_scheduler(
        worker.optimizer, SimpleNamespace(num_warmup_steps=0, weight_decay=0.01)
    )
    worker.scheduler.step(1)
    worker.actor_module = [
        SimpleNamespace(named_parameters=lambda: [("linear_proj.adapter.linear_in.weight", parameter)])
    ]
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    before = worker.describe_restorable_state()
    saved_adam = deepcopy(optimizer.optimizer.state_dict())
    moments_before = [optimizer.optimizer.state[parameter][key].clone() for key in ("exp_avg", "exp_avg_sq")]
    after = worker.mutate_restorable_state()
    assert all(after[key] != before[key] for key in ("model", "optimizer", "scheduler"))
    assert worker.scheduler.num_steps == 2
    for key, expected in zip(("exp_avg", "exp_avg_sq"), moments_before):
        assert not torch.equal(optimizer.optimizer.state[parameter][key], expected)
    optimizer.optimizer.load_state_dict(saved_adam)
    for key, expected in zip(("exp_avg", "exp_avg_sq"), moments_before):
        torch.testing.assert_close(optimizer.optimizer.state[parameter][key], expected, rtol=0, atol=0)


def test_dtype_tuple_state_keys_are_deterministic_and_type_safe():
    state = {0: {(torch.float32, torch.float32): [torch.ones(2)]}}
    assert fingerprint_state(state) == fingerprint_state(state)
    assert fingerprint_state(state) != fingerprint_state({0: {str((torch.float32, torch.float32)): [torch.ones(2)]}})


@pytest.mark.parametrize("stage", [0, 1, 2])
@pytest.mark.parametrize("defect", [None, "missing_side", "frozen", "missing_layer"])
def test_physical_factor_inventory_checks_every_layer_and_both_sides(monkeypatch, stage, defect):
    from examples.model_checks import h1_policy_audit

    class Layer(torch.nn.Module):
        def __init__(self, number):
            super().__init__()
            self.layer_number = number
            self.self_attention = torch.nn.Module()
            self.mlp = torch.nn.Module()
            parents = [self.self_attention]
            targets = [TARGET_MODULES[:5]]
            if number == 1:
                parents.append(self.mlp)
                targets.append(TARGET_MODULES[-2:])
            else:
                for category in ("experts", "shared_experts"):
                    module = torch.nn.Module()
                    self.mlp.add_module(category, module)
                    parents.append(module)
                    targets.append(TARGET_MODULES[-2:])
            for parent, suffixes in zip(parents, targets):
                for target in suffixes:
                    module = torch.nn.Module()
                    module.adapter = torch.nn.Module()
                    module.adapter.linear_in = torch.nn.Linear(2, 2, bias=False)
                    module.adapter.linear_out = torch.nn.Linear(2, 2, bias=False)
                    parent.add_module(target, module)

    chunk = torch.nn.Module()
    chunk.layers = torch.nn.ModuleList([Layer(stage * 2 + 1), Layer(stage * 2 + 2)])
    target = chunk.layers[0].self_attention.linear_q_down_proj
    if defect == "missing_side":
        del target.adapter.linear_out
    elif defect == "frozen":
        target.adapter.linear_out.weight.requires_grad_(False)
    elif defect == "missing_layer":
        for parent in chunk.layers[1].modules():
            if hasattr(parent, "adapter"):
                del parent.adapter
    worker = SimpleNamespace(actor_module=[chunk], provider=SimpleNamespace(moe_layer_freq=[0, 1, 1, 1, 1, 1]))
    monkeypatch.setattr(h1_policy_audit, "TransformerLayer", Layer)
    monkeypatch.setattr(h1_policy_audit.parallel_state, "get_pipeline_model_parallel_rank", lambda: stage)
    monkeypatch.setattr(h1_policy_audit.parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: stage * 8)
    result = H1PolicyAuditWorker.describe_lora_factors(worker)
    assert result["passed"] is (defect is None)
    assert result["layer_numbers"] == [stage * 2 + 1, stage * 2 + 2]
