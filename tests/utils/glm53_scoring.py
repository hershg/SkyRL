"""Score fixed GLM response tokens with controlled client submission."""

import asyncio


async def score_fixed_responses(client, prompts, responses, model, concurrent=True):
    async def score_response(prompt, response):
        result = await client.sample(
            {
                "json": {
                    "prompt": {"chunks": [{"tokens": prompt + response}]},
                    "num_samples": 1,
                    "sampling_params": {"temperature": 0.0, "max_tokens": 1},
                    "include_prompt_logprobs": True,
                    "model": model,
                }
            }
        )
        logprobs = result["prompt_logprobs"]
        assert logprobs is not None
        assert len(logprobs) == len(prompt) + len(response)
        response_logprobs = logprobs[len(prompt) :]
        assert all(logprob is not None for logprob in response_logprobs)
        return response_logprobs

    pairs = list(zip(prompts, responses, strict=True))
    if concurrent:
        return await asyncio.gather(*(score_response(p, r) for p, r in pairs))
    return [await score_response(p, r) for p, r in pairs]
