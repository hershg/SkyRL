"""CPU controls for LoRA checks; no server, GPU or process signals."""

from types import SimpleNamespace

import pytest
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from tests.tinker.model_checks import check_lora_capacity, check_lora_runtime


def make_datum(length: int, start: int = 10) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(list(range(start, start + length))),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.arange(start + 1, start + length + 1)),
            "weights": TensorData.from_torch(torch.ones(length)),
        },
    )


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class Trainer:
    def __init__(self, stale=False, reverse=False, batch_drift=0.0, fail_backward=0, nan_optimizer=False, rank=32):
        self.stale, self.reverse, self.batch_drift = stale, reverse, batch_drift
        self.fail_backward, self.nan_optimizer, self.rank = fail_backward, nan_optimizer, rank
        self.updates, self.backwards = 0, 0
        self.events = []

    def get_info(self):
        return SimpleNamespace(
            is_lora=self.rank > 0, lora_rank=self.rank, model_id="test-model", model_name="test-base"
        )

    def forward(self, data, loss):
        assert loss == "cross_entropy"
        outputs = [
            {
                "logprobs": TensorData.from_torch(
                    -datum.loss_fn_inputs["target_tokens"].to_torch().float() * 0.01
                    + self.updates * 0.25
                    + (self.batch_drift if len(data) > 1 else 0)
                )
            }
            for datum in data
        ]
        return Future(SimpleNamespace(metrics={}, loss_fn_outputs=list(reversed(outputs)) if self.reverse else outputs))

    def forward_backward(self, data, loss, config=None):
        self.events.append("backward")
        self.backwards += 1
        if self.backwards == self.fail_backward:
            raise RuntimeError("out of memory")
        if loss == "ppo":
            assert config == {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03}
            assert data[0].loss_fn_inputs["advantages"].to_torch().tolist() == [1.0] * len(
                data[0].model_input.to_ints()
            )
        outputs = [{"logprobs": TensorData.from_torch(torch.zeros(len(d.model_input.to_ints())))} for d in data]
        return Future(SimpleNamespace(metrics={"total_loss:sum": 1.0}, loss_fn_outputs=outputs))

    def optim_step(self, adam):
        self.events.append("optimizer")
        self.updates += 1
        return Future(SimpleNamespace(metrics={"skyrl.ai/grad_norm": float("nan") if self.nan_optimizer else 0.5}))

    def save_weights_and_get_sampling_client(self):
        self.events.append("publication")
        return Sampler(self, 0 if self.stale else self.updates)

    def save_state(self, name):
        self.events.append("checkpoint")
        return Future(SimpleNamespace(path=f"tinker://test-model/weights/{name}"))


class Sampler:
    def __init__(self, trainer, update):
        self.trainer, self.update = trainer, update

    def compute_logprobs(self, prompt):
        return Future([None] + [-token * 0.01 + self.update * 0.25 for token in prompt.to_ints()[1:]])

    def sample(self, prompt, num_samples, sampling_params):
        self.trainer.events.append("sample")
        assert len(prompt.to_ints()) == 7 and num_samples == sampling_params.max_tokens == 1
        return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[12], logprobs=[-0.5])]))


def run_runtime(trainer, report):
    return check_lora_runtime(
        trainer,
        32,
        [make_datum(3), make_datum(5, 30)],
        [1.0, -1.0],
        types.AdamParams(learning_rate=1e-4),
        mean_atol=0.3,
        update_atol=1e-5,
        batching_atol=1e-5,
        report=report,
    )


def run_capacity(trainer, report, batches=None, loss_fn="cross_entropy", loss_fn_config=None):
    return check_lora_capacity(
        trainer,
        32,
        batches or [[make_datum(8)], [make_datum(3), make_datum(5)]],
        8,
        loss_fn,
        loss_fn_config or {},
        types.AdamParams(learning_rate=1e-4),
        types.ModelInput.from_ints(list(range(7))),
        "step-2",
        report,
    )


def test_real_update_and_fixed_token_scores_agree():
    trainer, report = Trainer(), {}
    run_runtime(trainer, report)
    assert report["status"] == "passed_runtime"
    assert report["update"]["mean_abs"] < 1e-5
    assert trainer.events == ["publication", "backward", "optimizer", "publication"]


def test_stale_publication_fails_even_inside_absolute_tolerance():
    report = {}
    with pytest.raises(AssertionError, match="unresolved"):
        run_runtime(Trainer(stale=True), report)
    assert report["after"]["agreement"]["mean_abs"] < 0.3
    assert report["sampler_change"] == 0


@pytest.mark.parametrize("rank", [0, 8])
def test_full_weight_or_wrong_rank_client_is_not_lora_qualification(rank):
    trainer = Trainer(rank=rank)
    with pytest.raises(AssertionError, match="declared LoRA"):
        run_runtime(trainer, {})
    assert not trainer.events


@pytest.mark.parametrize("fault", [{"reverse": True}, {"batch_drift": 0.5}])
def test_bad_batch_order_or_scores_fail_before_update(fault):
    trainer = Trainer(**fault)
    with pytest.raises(AssertionError):
        run_runtime(trainer, {})
    assert trainer.updates == 0


def test_nonfinite_optimizer_fails_before_publication():
    trainer = Trainer(nan_optimizer=True)
    with pytest.raises(AssertionError, match="Nonfinite"):
        run_runtime(trainer, {})
    assert trainer.events.count("publication") == 1


def test_capacity_keeps_gradients_and_optimizer_state_across_two_cycles():
    trainer, report = Trainer(), {}
    run_capacity(trainer, report)
    assert trainer.events == ["backward", "backward", "optimizer", "publication", "sample"] * 2 + ["checkpoint"]
    assert report["batch_lengths"] == [[8], [3, 5]]
    assert report["checkpoint_path"] == "tinker://test-model/weights/step-2"
    assert report["status"] == "passed_capacity"


@pytest.mark.parametrize("failed_call", [2, 3, 4])
def test_capacity_detects_later_backward_oom_and_preserves_failed_phase(failed_call):
    trainer, report = Trainer(fail_backward=failed_call), {}
    with pytest.raises(RuntimeError, match="out of memory"):
        run_capacity(trainer, report)
    assert report["active_phase"] == f"step_{(failed_call - 1) // 2}/backward_{(failed_call - 1) % 2}"
    assert report["active_phase"] in report["seconds"]
    assert "checkpoint" not in trainer.events


@pytest.mark.parametrize("length", [7, 9])
def test_capacity_rejects_wrong_context_before_remote_work(length):
    trainer = Trainer()
    with pytest.raises(AssertionError):
        run_capacity(trainer, {}, [[make_datum(length)], [make_datum(3), make_datum(5)]])
    assert not trainer.events


def test_capacity_requires_all_exact_context_positions_to_be_scored():
    datum = make_datum(8)
    datum.loss_fn_inputs["weights"] = TensorData.from_torch(torch.tensor([0.0] + [1.0] * 7))
    with pytest.raises(AssertionError, match="Score every"):
        run_capacity(Trainer(), {}, [[datum], [make_datum(3), make_datum(5)]])


def test_capacity_runs_the_selected_ppo_path_not_a_cross_entropy_substitute():
    batches = [[make_datum(8)], [make_datum(3), make_datum(5)]]
    for batch in batches:
        for datum in batch:
            datum.loss_fn_inputs["advantages"] = datum.loss_fn_inputs.pop("weights")
            datum.loss_fn_inputs["logprobs"] = TensorData.from_torch(torch.zeros(len(datum.model_input.to_ints())))
    trainer, report = Trainer(), {}
    run_capacity(trainer, report, batches, "ppo", {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03})
    assert trainer.backwards == 4 and trainer.updates == 2
    assert report["status"] == "passed_capacity"
