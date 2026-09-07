"""Scoring budgets for actual-generation GLM attention-path probes."""

import numpy as np


def build_long_prefill_prompts(tokenizer, prompts, repetitions):
    reference = "Reference note: water freezes at zero degrees Celsius.\n" * repetitions
    tokens = tokenizer.apply_chat_template(
        [[{"role": "user", "content": reference + prompt}] for prompt in prompts],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    assert all(2048 < len(prompt) < 32768 - 128 for prompt in tokens)
    return tokens


def get_generation_logprob_metrics(sampler, trainer, mask):
    assert sampler.shape == trainer.shape == mask.shape
    mask = mask.astype(bool)
    assert mask.ndim == 2 and mask.any(axis=1).all()
    signed = sampler.astype(np.float64) - trainer.astype(np.float64)
    assert np.isfinite(signed[mask]).all()
    error = np.abs(signed[mask])
    return {
        "tokens": int(mask.sum()),
        "mean": float(error.mean()),
        "p99": float(np.quantile(error, 0.99)),
        "max": float(error.max()),
        "signed_mean": float(signed[mask].mean()),
        "sequence_signed_means": [float(row[valid].mean()) for row, valid in zip(signed, mask, strict=True)],
    }


def assert_generation_logprob_budget(metrics):
    assert metrics["mean"] < 0.075
    assert metrics["p99"] < 0.75
    assert metrics["max"] < 5.0
    assert abs(metrics["signed_mean"]) < 0.05
    assert max(abs(value) for value in metrics["sequence_signed_means"]) < 0.05


def compare_generation_scoring_paths(arrays):
    comparisons = {
        "trainer": ("sampler", "trainer"),
        "replay": ("sampler", "replay"),
        "decode_vs_teacher_forced": ("sampler", "teacher_forced_sampler"),
        "teacher_forced_vs_trainer": ("teacher_forced_sampler", "trainer"),
        "teacher_forced_vs_replay": ("teacher_forced_sampler", "teacher_forced_replay"),
    }
    return {
        label: get_generation_logprob_metrics(arrays[left], arrays[right], arrays["response_mask"])
        for label, (left, right) in comparisons.items()
    }
