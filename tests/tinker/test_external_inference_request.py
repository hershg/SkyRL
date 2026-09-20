import json
from functools import partial
from unittest.mock import create_autospec

import httpx
import pytest

from skyrl.tinker.api import SampleRequest
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.external_future_store import ExternalFutureStore
from skyrl.tinker.extra import external_inference


@pytest.mark.asyncio
@pytest.mark.parametrize("read_timeout", [300.0, 1800.0])
async def test_external_sampling_preserves_router_contract_and_read_timeout(monkeypatch, read_timeout):
    def respond(request):
        assert request.extensions["timeout"] == {
            "connect": 10.0,
            "read": read_timeout,
            "write": 300.0,
            "pool": 300.0,
        }
        payload = json.loads(request.content)
        if type(payload["logprobs"]) is not int:
            return httpx.Response(422, json={"detail": "logprobs must be an integer"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "token_ids": [7],
                        "logprobs": {"token_logprobs": [-0.5]},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    request = SampleRequest(
        prompt={"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
        sampling_params={"max_tokens": 1},
        base_model="test-model",
    )
    store = create_autospec(ExternalFutureStore, instance=True)
    config = EngineConfig(
        base_model="test-model",
        external_inference_url="http://example.test",
        forwarding_inference_timeout_sec=read_timeout,
    )
    http_client = partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    monkeypatch.setattr(external_inference.httpx, "AsyncClient", http_client)
    client = external_inference.ExternalInferenceClient(config, None, store)
    await client.call_and_store_result(1, request, "", "", base_model="test-model")
    store.complete.assert_awaited_once()
    request_id, result, status = store.complete.await_args.args
    assert request_id == 1 and status == RequestStatus.COMPLETED
    assert result.sequences[0].tokens == [7]
    assert result.sequences[0].logprobs == [-0.5]
