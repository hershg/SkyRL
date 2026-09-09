"""Model checks using caller-owned Tinker SDK clients and tokenized fixtures."""

import math
from contextlib import contextmanager
from time import monotonic

import torch
from tinker import types
from tinker.types.tensor_data import TensorData


@contextmanager
def _time_phase(report: dict, name: str):
    report["active_phase"] = name
    started = monotonic()
    try:
        yield
    finally:
        report.setdefault("seconds", {})[name] = monotonic() - started


def _assert_finite_metrics(metrics: dict) -> None:
    for value in metrics.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), "Nonfinite training/optimizer metric"


def _record_lora(trainer, expected_rank: int, report: dict) -> None:
    info = trainer.get_info()
    assert (
        info.is_lora is True and info.lora_rank == expected_rank
    ), "Expected the declared LoRA adapter, not full weights"
    report["model"] = {"id": info.model_id, "name": info.model_name, "lora_rank": info.lora_rank}


def _select_scores(rows: list, data: list[types.Datum], weight_key: str = "weights") -> torch.Tensor:
    assert len(rows) == len(data) and data
    selected = []
    for row, datum in zip(rows, data, strict=True):
        row = torch.as_tensor(row, dtype=torch.float64)
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights = datum.loss_fn_inputs[weight_key].to_torch()
        assert row.shape == targets.shape == weights.shape
        assert len(datum.model_input.to_ints()) == targets.numel()
        assert torch.isfinite(row).all() and torch.isfinite(weights).all()
        assert (weights != 0).any()
        selected.append(row[weights != 0])
    return torch.cat(selected)


def _get_training_rows(trainer, data: list[types.Datum]) -> list:
    result = trainer.forward(data, "cross_entropy").result()
    _assert_finite_metrics(result.metrics)
    rows = [output["logprobs"].to_torch() for output in result.loss_fn_outputs]
    _select_scores(rows, data)
    return rows


def _get_sampler_scores(sampler, data: list[types.Datum]) -> torch.Tensor:
    rows = []
    for datum in data:
        inputs = datum.model_input.to_ints()
        targets = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
        assert inputs[1:] == targets[:-1], "Fixture must use contiguous shifted tokens"
        scores = sampler.compute_logprobs(types.ModelInput.from_ints(inputs + targets[-1:])).result()
        assert len(scores) == len(inputs) + 1 and all(score is not None for score in scores[1:])
        rows.append(scores[1:])
    return _select_scores(rows, data)


def _compare_scores(left: torch.Tensor, right: torch.Tensor) -> dict:
    assert left.shape == right.shape and left.numel()
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    difference = (left - right).abs()
    return {"mean_abs": difference.mean().item(), "max_abs": difference.max().item()}


def check_lora_runtime(
    trainer,
    expected_rank: int,
    data: list[types.Datum],
    advantages: list[float],
    adam: types.AdamParams,
    mean_atol: float,
    update_atol: float,
    batching_atol: float,
    report: dict,
) -> dict:
    """Check fixed-token agreement, example ordering and publication of one real PPO update."""
    _record_lora(trainer, expected_rank, report)
    assert len(data) == len(advantages) and len(data) >= 2
    assert all(math.isfinite(value) for value in advantages)
    assert len({len(d.model_input.to_ints()) for d in data}) >= 2, "Use mixed-length examples"
    assert all(math.isfinite(x) and x >= 0 for x in (mean_atol, update_atol, batching_atol))
    report.update(status="in_progress", tolerances={"mean": mean_atol, "update": update_atol, "batch": batching_atol})
    snapshots = []
    for phase in ("before", "after"):
        with _time_phase(report, phase + "/trainer"):
            rows = _get_training_rows(trainer, data)
            training = _select_scores(rows, data)
            individual = [_get_training_rows(trainer, [datum])[0] for datum in reversed(data)]
            batching = _compare_scores(training, _select_scores(list(reversed(individual)), data))
        report[phase] = {"batching": batching, "trainer": training.tolist()}
        assert batching["mean_abs"] <= batching_atol, "Batched scores differ from original-example order"
        with _time_phase(report, phase + "/publication"):
            sampler = trainer.save_weights_and_get_sampling_client()
        with _time_phase(report, phase + "/sampler"):
            sampling = _get_sampler_scores(sampler, data)
            repeat = _get_sampler_scores(sampler, data)
        agreement = _compare_scores(training, sampling)
        report[phase].update(sampler=sampling.tolist(), agreement=agreement, repeat=_compare_scores(sampling, repeat))
        assert agreement["mean_abs"] <= mean_atol, "Trainer/sampler scores disagree on identical tokens"
        snapshots.append((training, sampling))
        if phase == "before":
            ppo_data = [
                types.Datum(
                    model_input=datum.model_input,
                    loss_fn_inputs={
                        "target_tokens": datum.loss_fn_inputs["target_tokens"],
                        "logprobs": TensorData.from_torch(reference),
                        "advantages": TensorData.from_torch(datum.loss_fn_inputs["weights"].to_torch() * advantage),
                    },
                )
                for datum, reference, advantage in zip(data, rows, advantages, strict=True)
            ]
            with _time_phase(report, "backward"):
                backward = trainer.forward_backward(
                    ppo_data, "ppo", {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03}
                ).result()
                _assert_finite_metrics(backward.metrics)
                _select_scores([output["logprobs"].to_torch() for output in backward.loss_fn_outputs], data)
                report["backward_metrics"] = backward.metrics
            with _time_phase(report, "optimizer"):
                optimizer = trainer.optim_step(adam).result()
                _assert_finite_metrics(optimizer.metrics)
                report["optimizer_metrics"] = optimizer.metrics
    trainer_delta = snapshots[1][0] - snapshots[0][0]
    sampler_delta = snapshots[1][1] - snapshots[0][1]
    report["update"] = _compare_scores(trainer_delta, sampler_delta)
    report["trainer_change"] = trainer_delta.abs().max().item()
    report["sampler_change"] = sampler_delta.abs().max().item()
    noise = max(report[phase]["repeat"]["max_abs"] for phase in ("before", "after"))
    assert (
        report["trainer_change"] > 0 and report["sampler_change"] > noise
    ), "Update unresolved; fixture is inconclusive"
    assert report["update"]["mean_abs"] <= update_atol, "Sampler did not reflect the trainer update"
    report.update(status="passed_runtime", active_phase=None)
    return report


def check_lora_capacity(
    trainer,
    expected_rank: int,
    batches: list[list[types.Datum]],
    context_length: int,
    loss_fn: str,
    loss_fn_config: dict,
    adam: types.AdamParams,
    sample_prompt: types.ModelInput,
    checkpoint_name: str,
    report: dict,
) -> dict:
    """Exercise exact-context accumulation and two optimizer/publication cycles without resetting the model."""
    _record_lora(trainer, expected_rank, report)
    assert context_length > 0 and len(batches) == 2 and all(batches)
    weight_key = {"cross_entropy": "weights", "ppo": "advantages"}[loss_fn]
    assert len(sample_prompt.to_ints()) == context_length - 1, "One generated token must reach the declared context"
    data = [datum for batch in batches for datum in batch]
    lengths = [len(d.model_input.to_ints()) for d in data]
    assert max(lengths) == context_length and len(set(lengths)) >= 2
    for datum, length in zip(data, lengths, strict=True):
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        weights = datum.loss_fn_inputs[weight_key].to_torch()
        assert targets.numel() == weights.numel() == length
        assert torch.isfinite(weights).all()
        assert datum.model_input.to_ints()[1:] == targets.tolist()[:-1]
        if loss_fn == "ppo":
            reference = datum.loss_fn_inputs["logprobs"].to_torch()
            assert reference.shape == targets.shape and torch.isfinite(reference).all()
        if length == context_length:
            assert torch.count_nonzero(weights) == context_length, "Score every exact-context input position"
    report.update(
        status="in_progress",
        context_length=context_length,
        batch_lengths=[[len(d.model_input.to_ints()) for d in b] for b in batches],
    )
    for step in range(2):
        for index, batch in enumerate(batches):
            with _time_phase(report, f"step_{step}/backward_{index}"):
                result = trainer.forward_backward(batch, loss_fn, loss_fn_config).result()
                _assert_finite_metrics(result.metrics)
                _select_scores([output["logprobs"].to_torch() for output in result.loss_fn_outputs], batch, weight_key)
        with _time_phase(report, f"step_{step}/optimizer"):
            result = trainer.optim_step(adam).result()
            _assert_finite_metrics(result.metrics)
            report[f"optimizer_{step}"] = result.metrics
        with _time_phase(report, f"step_{step}/publication"):
            sampler = trainer.save_weights_and_get_sampling_client()
        with _time_phase(report, f"step_{step}/sample"):
            result = sampler.sample(
                sample_prompt, num_samples=1, sampling_params=types.SamplingParams(max_tokens=1, temperature=0)
            ).result()
            assert len(result.sequences) == 1 and len(result.sequences[0].tokens) == 1
            assert len(result.sequences[0].logprobs) == 1 and math.isfinite(result.sequences[0].logprobs[0])
    with _time_phase(report, "checkpoint"):
        checkpoint = trainer.save_state(checkpoint_name).result()
        assert checkpoint.path
        report["checkpoint_path"] = checkpoint.path
    report.update(status="passed_capacity", active_phase=None)
    return report
