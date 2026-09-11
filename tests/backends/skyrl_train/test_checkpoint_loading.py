import tarfile
from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.workers.worker import Worker
from skyrl.backends.skyrl_train_backend import (
    MegatronBackendOverrides,
    SkyRLTrainBackend,
)


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_load_checkpoint_restores_requested_training_state(tmp_path, load_optimizer):
    checkpoint_path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(checkpoint_path, "w"):
        pass

    backend = object.__new__(SkyRLTrainBackend)
    backend._model_ids_to_role = {"model_test": "policy"}
    backend._dispatch = MagicMock()

    backend.load_checkpoint(str(checkpoint_path), "model_test", load_optimizer=load_optimizer)

    backend._dispatch.load_checkpoint.assert_called_once()
    call = backend._dispatch.load_checkpoint.call_args
    assert call.kwargs["load_optimizer_states"] is load_optimizer
    assert call.kwargs["load_lr_scheduler_states"] is load_optimizer


def test_weights_only_load_does_not_reset_live_optimizer():
    worker = object.__new__(Worker)
    worker.model = MagicMock()
    worker.optimizer = MagicMock()
    worker.scheduler = MagicMock()
    worker.strategy = MagicMock()
    worker.strategy.load_checkpoint.return_value = ("/checkpoint", {})
    live_optimizer = worker.optimizer
    live_scheduler = worker.scheduler

    Worker.load_checkpoint(
        worker,
        "/checkpoint",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    assert worker.optimizer is live_optimizer
    assert worker.scheduler is live_scheduler

    worker.strategy.load_checkpoint.assert_called_once_with(
        model=worker.model,
        optimizer=None,
        scheduler=None,
        ckpt_dir="/checkpoint",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )


def test_build_policy_requests_configured_node_resource():
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(policy_node_resource="trainer_node")
    backend._cfg = MagicMock()
    backend._cfg.trainer.placement.colocate_all = False
    backend._cfg.trainer.placement.policy_num_nodes = 1
    backend._cfg.trainer.placement.policy_num_gpus_per_node = 8
    backend._cfg.trainer.policy.model.lora.rank = 0
    backend._colocate_pg = None
    backend._tokenizer = MagicMock(pad_token_id=0)
    backend._inference_engine_client = None

    with (
        patch("skyrl.backends.skyrl_train_backend.PPORayActorGroup") as actor_group,
        patch("skyrl.backends.skyrl_train_backend.WorkerDispatch"),
        patch("skyrl.backends.skyrl_train_backend.ray.get"),
    ):
        backend._build_policy(MagicMock(), "model_test")

    assert actor_group.call_args.kwargs["resources"] == {"trainer_node": 1.0}
    assert actor_group.call_args.kwargs["num_resources_per_node"] == 8


def test_policy_node_resource_rejects_colocated_policy():
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(policy_node_resource="trainer_node")
    backend._cfg = MagicMock()
    backend._cfg.trainer.placement.colocate_all = True
    backend._cfg.trainer.policy.model.lora.rank = 0
    backend._colocate_pg = MagicMock()

    with pytest.raises(ValueError, match="requires trainer.placement.colocate_all=False"):
        backend._build_policy(MagicMock(), "model_test")
