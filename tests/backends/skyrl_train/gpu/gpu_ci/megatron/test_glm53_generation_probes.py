import numpy as np
import pytest

from tests.backends.skyrl_train.gpu.gpu_ci.megatron.glm53_generation_probes import (
    assert_generation_logprob_budget,
    compare_generation_scoring_paths,
    get_generation_logprob_metrics,
)


def test_generation_parity_ignores_only_masked_positions():
    sampler = np.array([[-0.2, -0.4, 99], [-0.3, 88, 77]])
    trainer = np.array([[-0.21, -0.39, 0], [-0.31, 0, 0]])
    mask = np.array([[1, 1, 0], [1, 0, 0]])
    metrics = get_generation_logprob_metrics(sampler, trainer, mask)
    assert metrics["tokens"] == 3
    assert metrics["mean"] == pytest.approx(0.01)
    assert_generation_logprob_budget(metrics)


def test_generation_parity_rejects_per_sequence_bias_hidden_by_batch_mean():
    sampler = np.array([[0.06, 0.06], [-0.06, -0.06]])
    metrics = get_generation_logprob_metrics(sampler, np.zeros_like(sampler), np.ones_like(sampler))
    assert metrics["signed_mean"] == 0
    assert metrics["mean"] < 0.075
    with pytest.raises(AssertionError):
        assert_generation_logprob_budget(metrics)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1.0])
def test_generation_parity_rejects_invalid_scores(value):
    sampler = np.array([[value]])
    with pytest.raises(AssertionError):
        metrics = get_generation_logprob_metrics(sampler, np.zeros_like(sampler), np.ones_like(sampler))
        assert_generation_logprob_budget(metrics)


def test_generation_parity_rejects_missing_sequence_scores():
    values = np.zeros((2, 3))
    with pytest.raises(AssertionError):
        get_generation_logprob_metrics(values, values, np.array([[1, 1, 1], [0, 0, 0]]))
    with pytest.raises(AssertionError):
        get_generation_logprob_metrics(values, values[:, :2], np.ones_like(values))


def test_scoring_comparisons_do_not_conflate_decode_and_teacher_forced_routes():
    arrays = {
        "sampler": np.array([[-0.1, -0.2]]),
        "trainer": np.array([[-0.2, -0.3]]),
        "replay": np.array([[-0.11, -0.21]]),
        "teacher_forced_sampler": np.array([[-0.15, -0.25]]),
        "teacher_forced_replay": np.array([[-0.17, -0.27]]),
        "response_mask": np.ones((1, 2)),
    }
    metrics = compare_generation_scoring_paths(arrays)
    assert metrics["trainer"]["mean"] == pytest.approx(0.1)
    assert metrics["replay"]["mean"] == pytest.approx(0.01)
    assert metrics["decode_vs_teacher_forced"]["mean"] == pytest.approx(0.05)
    assert metrics["teacher_forced_vs_trainer"]["mean"] == pytest.approx(0.05)
    assert metrics["teacher_forced_vs_replay"]["mean"] == pytest.approx(0.02)
