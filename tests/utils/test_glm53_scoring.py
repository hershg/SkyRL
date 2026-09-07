import asyncio

import pytest

from tests.utils.glm53_scoring import score_fixed_responses


class RecordingClient:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.requests = []

    async def sample(self, request):
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0)
            tokens = request["json"]["prompt"]["chunks"][0]["tokens"]
            return {"prompt_logprobs": [None] + [-token / 100 for token in tokens[1:]]}
        finally:
            self.active -= 1


@pytest.mark.parametrize("concurrent,expected_peak", [(False, 1), (True, 4)])
def test_submission_mode_changes_overlap_without_changing_scored_work(
    concurrent, expected_peak
):
    client = RecordingClient()
    prompts = [[1, 2], [3], [4, 5, 6], [7, 8]]
    responses = [[9], [10, 11], [12], [13, 14, 15]]
    actual = asyncio.run(
        score_fixed_responses(client, prompts, responses, "test-adapter", concurrent)
    )
    assert actual == [[-token / 100 for token in response] for response in responses]
    assert client.peak == expected_peak and client.active == 0
    assert [r["json"]["prompt"]["chunks"][0]["tokens"] for r in client.requests] == [
        p + r for p, r in zip(prompts, responses, strict=True)
    ]
    assert all(r["json"]["model"] == "test-adapter" for r in client.requests)
    assert prompts == [[1, 2], [3], [4, 5, 6], [7, 8]]
    assert responses == [[9], [10, 11], [12], [13, 14, 15]]


def test_concurrent_completion_keeps_original_response_order():
    async def run():
        second_finished = asyncio.Event()
        completed = []

        class Client:
            async def sample(self, request):
                tokens = request["json"]["prompt"]["chunks"][0]["tokens"]
                if tokens[0] == 1:
                    await second_finished.wait()
                else:
                    second_finished.set()
                completed.append(tokens[0])
                return {"prompt_logprobs": [None, -tokens[-1] / 100]}

        result = await asyncio.wait_for(
            score_fixed_responses(Client(), [[1], [2]], [[3], [4]], "test-adapter"),
            timeout=1,
        )
        assert completed == [2, 1]
        assert result == [[-0.03], [-0.04]]

    asyncio.run(run())


@pytest.mark.parametrize("logprobs", [None, [None], [None, None], [None, -0.1, -0.2]])
def test_scoring_rejects_missing_or_misaligned_response_logprobs(logprobs):
    class Client:
        async def sample(self, request):
            return {"prompt_logprobs": logprobs}

    with pytest.raises(AssertionError):
        asyncio.run(score_fixed_responses(Client(), [[1]], [[2]], "test-adapter"))
