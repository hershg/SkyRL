import pytest

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRardtProducerRendezvous,
    LoRardtRollbackError,
    LoRardtServerLifecycle,
    LoRAUpdateRequest,
)


class _Engine:
    def __init__(self, fail_methods=()):
        self.calls = []
        self.fail_methods = set(fail_methods)

    async def collective_rpc(self, method, kwargs):
        self.calls.append((method, kwargs))
        if method in self.fail_methods:
            raise RuntimeError(f"forced {method} failure")


def _rendezvous():
    key = "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight"
    layout = LoRABridgeSourceLayout(
        "adapter",
        (
            LoRABridgeSource(
                key=key,
                source_rank=0,
                hf_param_names=("down_proj.lora_B.weight",),
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
    return LoRardtProducerRendezvous(layout, ((0, "lora-rdt-0"),), 1)


def _request(generation=1):
    return LoRAUpdateRequest.from_json_dict(
        {
            "adapter_name": "adapter",
            "generation": generation,
            "layout_digest": _rendezvous().layout.layout_digest,
            "source_dtype": "float32",
        }
    )


@pytest.mark.asyncio
async def test_server_lifecycle_commits_only_an_activated_generation():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    lifecycle._active_ids["adapter"] = 3

    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    with pytest.raises(ValueError, match="has not been activated"):
        await lifecycle.commit(engine, _request(), 4)
    assert lifecycle.get_active_adapter_id("adapter") == 3

    await lifecycle.activate(engine, _request(), 4)
    assert await lifecycle.commit(engine, _request(), 4)
    assert lifecycle.get_active_adapter_id("adapter") == 4
    assert [call[0] for call in engine.calls] == [
        "stage_lora_rdt_adapter",
        "activate_lora_rdt_adapter",
        "remove_lora_rdt_adapter",
    ]


@pytest.mark.asyncio
async def test_server_lifecycle_rolls_back_without_replaying_cleanup():
    lifecycle = LoRardtServerLifecycle()
    lifecycle._active_ids["adapter"] = 3
    engine = _Engine(fail_methods={"activate_lora_rdt_adapter"})
    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})

    with pytest.raises(RuntimeError, match="forced"):
        await lifecycle.activate(engine, _request(), 4)
    assert await lifecycle.rollback(engine, _request(), 4)
    assert not await lifecycle.rollback(engine, _request(), 4)

    assert lifecycle.get_active_adapter_id("adapter") == 3
    assert [call[0] for call in engine.calls] == [
        "stage_lora_rdt_adapter",
        "activate_lora_rdt_adapter",
        "restore_lora_rdt_adapter",
        "discard_lora_rdt_adapter",
    ]


@pytest.mark.asyncio
async def test_terminal_replay_preserves_a_newer_staging_transaction():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    await lifecycle.activate(engine, _request(), 4)
    assert await lifecycle.commit(engine, _request(), 4)
    await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})
    before = list(engine.calls)

    assert not await lifecycle.commit(engine, _request(), 4)
    with pytest.raises(ValueError, match="already committed"):
        await lifecycle.rollback(engine, _request(), 4)
    with pytest.raises(ValueError, match="does not match"):
        await lifecycle.rollback(engine, _request(), 5)

    assert engine.calls == before
    assert lifecycle.get_active_adapter_id("adapter") == 4
    assert lifecycle._staged["adapter"][:2] == (_request(2), 5)


@pytest.mark.asyncio
async def test_failed_collective_stage_cleans_successful_ranks_before_retry():
    class PartialStageEngine(_Engine):
        def __init__(self):
            super().__init__()
            self.staged = {}
            self.fail_stage = True

        async def collective_rpc(self, method, kwargs):
            await super().collective_rpc(method, kwargs)
            if method == "stage_lora_rdt_adapter":
                self.staged[0] = kwargs["adapter_id"]
                if self.fail_stage:
                    raise RuntimeError("rank 1 failed staging")
                self.staged[1] = kwargs["adapter_id"]
            elif method == "discard_lora_rdt_adapter":
                self.staged = {
                    rank: adapter_id for rank, adapter_id in self.staged.items() if adapter_id != kwargs["adapter_id"]
                }

    engine = PartialStageEngine()
    lifecycle = LoRardtServerLifecycle()
    lifecycle._active_ids["adapter"] = 3

    with pytest.raises(RuntimeError, match="rank 1 failed"):
        await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})

    assert engine.staged == {}
    assert lifecycle.get_active_adapter_id("adapter") == 3
    assert lifecycle._staged == {}

    engine.fail_stage = False
    await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})

    assert engine.staged == {0: 5, 1: 5}
    assert lifecycle.get_active_adapter_id("adapter") == 3


@pytest.mark.asyncio
async def test_partial_stage_cleanup_failure_is_explicit():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine(fail_methods={"stage_lora_rdt_adapter", "discard_lora_rdt_adapter"})

    with pytest.raises(LoRardtRollbackError, match="could not discard partially staged"):
        await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})


@pytest.mark.asyncio
async def test_unload_releases_active_buffer_once_and_rejects_late_publication():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)

    await lifecycle.unload(engine, "adapter")
    calls = list(engine.calls)
    await lifecycle.unload(engine, "adapter")
    with pytest.raises(ValueError, match="unloaded"):
        await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})

    assert engine.calls == calls
    assert calls[-1] == ("remove_lora_rdt_adapter", {"adapter_id": 4})
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_failed_unload_can_retry_without_admitting_another_generation():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)
    engine.fail_methods.add("remove_lora_rdt_adapter")

    with pytest.raises(RuntimeError, match="forced"):
        await lifecycle.unload(engine, "adapter")
    with pytest.raises(ValueError, match="unloaded"):
        await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})

    assert lifecycle.get_active_adapter_id("adapter") == 4
    engine.fail_methods.clear()
    await lifecycle.unload(engine, "adapter")
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_unload_requires_pending_replacement_to_finish_before_removing_buffers():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    before = list(engine.calls)

    with pytest.raises(ValueError, match="unfinished replacement"):
        await lifecycle.unload(engine, "adapter")

    assert engine.calls == before
    await lifecycle.rollback(engine, _request(), 4)
    await lifecycle.unload(engine, "adapter")
    assert lifecycle.get_active_adapter_id("adapter") is None


@pytest.mark.asyncio
async def test_abort_before_delayed_stage_prevents_generation_revival():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    assert not await lifecycle.abort(engine, _request())
    with pytest.raises(ValueError, match="completed transaction"):
        await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    assert engine.calls == []
    await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})
    before = list(engine.calls)
    assert not await lifecycle.abort(engine, _request())
    assert engine.calls == before
    await lifecycle.activate(engine, _request(2), 5)
    assert lifecycle.get_active_adapter_id("adapter") == 5


@pytest.mark.asyncio
async def test_abort_preserves_committed_and_mismatched_generations():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    await lifecycle.activate(engine, _request(), 4)
    await lifecycle.commit(engine, _request(), 4)
    with pytest.raises(ValueError, match="already committed"):
        await lifecycle.abort(engine, _request())
    await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})
    before = list(engine.calls)
    mismatch = LoRAUpdateRequest.from_json_dict(
        {
            **_request(2).to_json_dict(),
            "layout_digest": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="not staged"):
        await lifecycle.abort(engine, mismatch)
    assert engine.calls == before
    assert lifecycle.get_active_adapter_id("adapter") == 4


@pytest.mark.asyncio
async def test_failed_stage_retains_cleanup_id_without_becoming_activatable():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine(fail_methods={"stage_lora_rdt_adapter", "discard_lora_rdt_adapter"})
    with pytest.raises(LoRardtRollbackError):
        await lifecycle.stage(engine, _rendezvous(), _request(), 4, {"r": 2})
    before = list(engine.calls)
    with pytest.raises(ValueError, match="did not finish staging"):
        await lifecycle.activate(engine, _request(), 4)
    assert engine.calls == before
    engine.fail_methods.clear()
    assert await lifecycle.abort(engine, _request())
    assert engine.calls[-1] == ("discard_lora_rdt_adapter", {"adapter_id": 4})
    assert not await lifecycle.abort(engine, _request())
    await lifecycle.stage(engine, _rendezvous(), _request(2), 5, {"r": 2})
