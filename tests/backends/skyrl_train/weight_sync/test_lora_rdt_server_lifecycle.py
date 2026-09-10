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

    async def pause_generation(self, mode):
        self.calls.append(("pause", mode))

    async def collective_rpc(self, method, kwargs):
        self.calls.append((method, kwargs))
        if method in self.fail_methods:
            raise RuntimeError(f"forced {method} failure")

    async def resume_generation(self):
        self.calls.append(("resume",))


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
async def test_server_lifecycle_drains_stages_activates_and_retires_old_adapter():
    lifecycle = LoRardtServerLifecycle()
    engine = _Engine()
    lifecycle._active_ids["adapter"] = 3

    assert await lifecycle.replace(engine, _rendezvous(), _request(), 4, {"r": 2}) == 4
    assert lifecycle.get_active_adapter_id("adapter") == 4
    assert [call[0] for call in engine.calls] == [
        "pause",
        "stage_lora_rdt_adapter",
        "activate_lora_rdt_adapter",
        "remove_lora_rdt_adapter",
        "resume",
    ]
    assert engine.calls[0] == ("pause", "wait")
    assert engine.calls[1][1]["adapter_id"] == 4


@pytest.mark.asyncio
async def test_server_lifecycle_rolls_back_staged_adapter_before_resuming():
    lifecycle = LoRardtServerLifecycle()
    lifecycle._active_ids["adapter"] = 3
    engine = _Engine(fail_methods={"activate_lora_rdt_adapter"})

    with pytest.raises(RuntimeError, match="forced"):
        await lifecycle.replace(engine, _rendezvous(), _request(), 4, {"r": 2})

    assert lifecycle.get_active_adapter_id("adapter") == 3
    assert [call[0] for call in engine.calls] == [
        "pause",
        "stage_lora_rdt_adapter",
        "activate_lora_rdt_adapter",
        "restore_lora_rdt_adapter",
        "discard_lora_rdt_adapter",
        "resume",
    ]


@pytest.mark.asyncio
async def test_server_lifecycle_leaves_engine_paused_when_rollback_fails():
    lifecycle = LoRardtServerLifecycle()
    lifecycle._active_ids["adapter"] = 3
    engine = _Engine(
        fail_methods={"activate_lora_rdt_adapter", "restore_lora_rdt_adapter"}
    )

    with pytest.raises(LoRardtRollbackError, match="remains paused"):
        await lifecycle.replace(engine, _rendezvous(), _request(), 4, {"r": 2})

    assert "resume" not in [call[0] for call in engine.calls]
