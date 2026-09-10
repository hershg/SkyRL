import pytest

from skyrl.backends.skyrl_train.weight_sync.lora_rdt.fleet_control import (
    LoRardtFleetTransaction,
)


@pytest.mark.asyncio
async def test_fleet_transaction_restores_every_server_after_activation_failure():
    events = []

    async def stage(url):
        events.append(("stage", url))

    async def pause():
        events.append(("pause",))

    async def activate(url):
        events.append(("activate", url))
        if url == "b":
            raise RuntimeError("forced activation failure")

    async def rollback(url):
        events.append(("rollback", url))

    async def commit(url):
        events.append(("commit", url))

    async def resume():
        events.append(("resume",))

    with pytest.raises(RuntimeError, match="forced activation failure"):
        await LoRardtFleetTransaction(["a", "b"]).replace(
            stage, pause, activate, rollback, commit, resume
        )

    assert events == [
        ("stage", "a"),
        ("stage", "b"),
        ("pause",),
        ("activate", "a"),
        ("activate", "b"),
        ("rollback", "a"),
        ("rollback", "b"),
        ("resume",),
    ]


@pytest.mark.asyncio
async def test_fleet_transaction_does_not_pause_when_staging_fails():
    events = []

    async def stage(url):
        events.append(("stage", url))
        if url == "b":
            raise RuntimeError("forced stage failure")

    async def impossible(*_):
        raise AssertionError("phase should not run")

    async def rollback(url):
        events.append(("rollback", url))

    with pytest.raises(RuntimeError, match="forced stage failure"):
        await LoRardtFleetTransaction(["a", "b"]).replace(
            stage, impossible, impossible, rollback, impossible, impossible
        )

    assert events == [
        ("stage", "a"),
        ("stage", "b"),
        ("rollback", "a"),
        ("rollback", "b"),
    ]
