"""CPU checks for the real-GSPO model-check primitives."""

from types import SimpleNamespace

import pytest
import torch

from examples.model_checks import real_gspo


class Batch(dict):
    @property
    def batch_size(self):
        return self["response_mask"].shape[0]

    def __getitem__(self, key):
        if isinstance(key, int):
            return Batch({name: value[key : key + 1].clone() for name, value in self.items()})
        return super().__getitem__(key)


def test_optimizer_batch_uses_current_scores_on_one_heterogeneous_sample():
    batch = Batch(
        response_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        action_log_probs=torch.zeros(2, 3),
        advantages=torch.zeros(2, 3),
    )
    update = real_gspo.build_optimizer_batch(batch, [-1, -2, -3, -4, -5], sample_index=1)
    assert update["action_log_probs"].tolist() == [[-3, -4, -5]]
    assert update["advantages"].tolist() == [[1, 1, 1]]


@pytest.mark.parametrize(
    "metrics,audits,norms,error",
    [
        ({"loss": 1.0}, [{"passed": True}], [2.0], None),
        ({}, [{"passed": True}], [2.0], "missing or nonfinite metrics"),
        (
            {"loss": float("nan")},
            [{"passed": True}],
            [2.0],
            "missing or nonfinite metrics",
        ),
        ({"loss": 1.0}, [{"passed": False}], [2.0], "per-rank LoRA gradients"),
        ({"loss": 1.0}, [{"passed": True}], [0.0], "gradient norm"),
    ],
)
def test_optimizer_update_fails_closed(monkeypatch, metrics, audits, norms, error):
    ray = pytest.importorskip("ray")
    from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput

    calls = []

    def dispatch(_, method, **kwargs):
        calls.append((method, kwargs))
        if method == "forward_backward":
            return "forward"
        if method == "describe_lora_gradients":
            return [{**audit, "rank": rank} for rank in range(8) for audit in audits]
        if method == "optim_step":
            return norms * 8
        if method == "describe_optimizer_state":
            return [{"passed": True, "rank": rank} for rank in range(8)]
        raise AssertionError(method)

    policy = SimpleNamespace(actor_infos=[], async_run_ray_method=dispatch)
    output = SimpleNamespace(metrics=metrics)
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(WorkerOutput, "cat", lambda *args: output)
    if error is not None:
        with pytest.raises(ValueError, match=error):
            real_gspo.run_optimizer_update(policy, "batch")
    else:
        result = real_gspo.run_optimizer_update(policy, "batch")
        assert result["grad_norms"] == [2.0] * 8
    assert calls[0] == (
        "forward_backward",
        {"data": "batch", "loss_fn": "gspo", "return_per_token_outputs": False},
    )


def test_heldout_comparison_uses_requested_variable_length_sample():
    result = real_gspo.compare_sample(
        [-1.0, -2.0, -3.0, -4.0, -5.0],
        [-9.0, -8.0, -3.1, -4.1, -5.1],
        [2, 3],
        1,
    )
    assert result["tokens"] == 3
    assert result["mean_abs"] == pytest.approx(0.1)


@pytest.mark.parametrize("fault", [None, "missing_stage", "duplicate_rank", "out_of_range"])
def test_rank_receipts_cover_all_twenty_four_pipeline_ranks(fault):
    receipts = [{"rank": rank} for rank in range(24)]
    if fault == "missing_stage":
        receipts = receipts[:16]
    elif fault == "duplicate_rank":
        receipts[-1] = {"rank": 0}
    elif fault == "out_of_range":
        receipts[-1] = {"rank": 24}
    if fault is None:
        assert real_gspo.validate_rank_receipts(list(reversed(receipts)), 24) == receipts
    else:
        with pytest.raises(ValueError):
            real_gspo.validate_rank_receipts(receipts, 24)
