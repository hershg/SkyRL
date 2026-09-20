import json

import httpx
import pytest

from skyrl.tinker.api import SampleRequest
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.external_inference import ExternalInferenceClient


@pytest.mark.asyncio
async def test_external_sampling_sends_router_compatible_logprobs():
    def respond(request):
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
        num_samples=1,
        base_model="test-model",
    )
    client = ExternalInferenceClient(
        EngineConfig(base_model="test-model", external_inference_url="http://example.test"), None
    )
    async with httpx.AsyncClient(base_url="http://example.test/v1", transport=httpx.MockTransport(respond)) as http:
        result = await client._forward_to_engine(request, "", "", http, base_model="test-model")
    assert result.sequences[0].tokens == [7]
    assert result.sequences[0].logprobs == [-0.5]
