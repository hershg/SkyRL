import asyncio
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRardtProducerRendezvous,
    LoRAUpdateRequest,
)


def _create_rendezvous():
    source = LoRABridgeSource(
        key="adapter.weight",
        source_rank=0,
        hf_param_names=("adapter.weight",),
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
    )
    layout = LoRABridgeSourceLayout("adapter", (source,))
    return LoRardtProducerRendezvous(layout, ((0, "producer-0"),), 1)


@pytest.mark.asyncio
async def test_terminal_http_replays_preserve_restored_route_and_newer_transaction():
    app = FastAPI()
    calls = []

    async def collective_rpc(method, kwargs):
        calls.append((method, kwargs))

    VLLMServerActor._add_custom_endpoints(app, SimpleNamespace(collective_rpc=collective_rpc), Namespace())
    adapter_ids = iter(range(10, 20))
    models = SimpleNamespace(
        lora_requests={},
        lora_resolver_lock=defaultdict(asyncio.Lock),
        lora_id_counter=SimpleNamespace(inc=lambda amount: next(adapter_ids)),
    )
    app.state.openai_serving_models = models
    rendezvous = _create_rendezvous()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://example.com") as client:

        async def stage(generation):
            body = {
                "lora_name": "adapter",
                "rendezvous": rendezvous.to_json_dict(),
                "request": LoRAUpdateRequest.from_layout(rendezvous.layout, generation).to_json_dict(),
                "adapter_config": {"r": 2},
            }
            response = await client.post("/skyrl/v1/stage_lora_rdt_adapter", json=body)
            assert response.status_code == 200
            return {**body, "adapter_id": response.json()["lora_int_id"]}

        async def phase(name, body):
            return await client.post(f"/skyrl/v1/{name}_lora_rdt_adapter", json=body)

        initial = await stage(0)
        assert (await phase("activate", initial)).status_code == 200
        assert (await phase("commit", initial)).status_code == 200
        original = models.lora_requests["adapter"]

        failed = await stage(1)
        assert (await phase("activate", failed)).status_code == 200
        assert (await phase("rollback", failed)).status_code == 200
        assert models.lora_requests["adapter"] is original
        calls_after_rollback = list(calls)
        assert (await phase("rollback", failed)).status_code == 200
        assert calls == calls_after_rollback
        assert models.lora_requests["adapter"] is original

        replacement = await stage(2)
        previous_requests = dict(models._skyrl_lora_rdt_previous_requests)
        assert (await phase("rollback", failed)).status_code == 200
        assert models._skyrl_lora_rdt_previous_requests == previous_requests
        assert (await phase("activate", replacement)).status_code == 200
        assert (await phase("commit", replacement)).status_code == 200
        new_route = models.lora_requests["adapter"]

        later = await stage(3)
        calls_before_replay = list(calls)
        previous_requests = dict(models._skyrl_lora_rdt_previous_requests)
        assert (await phase("commit", replacement)).status_code == 200
        assert calls == calls_before_replay
        assert models._skyrl_lora_rdt_previous_requests == previous_requests
        assert models.lora_requests["adapter"] is new_route

        assert (await phase("rollback", replacement)).status_code == 500
        assert (await phase("rollback", failed)).status_code == 500
        assert models.lora_requests["adapter"] is new_route
        assert models._skyrl_lora_rdt_previous_requests == previous_requests
        assert models._skyrl_lora_rdt_lifecycle._staged["adapter"][1] == later["adapter_id"]
