"""Exact token fixtures for paired GLM diagnostic runs."""

import json
from pathlib import Path


def load_fixed_responses(path: Path, prompts: list[list[int]]) -> list[list[int]]:
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    responses_by_prompt = {}
    for datum in payload["datums"]:
        prompt, response = datum["prompt_token_ids"], datum["response_token_ids"]
        assert prompt and response
        assert all(type(token) is int and token >= 0 for token in prompt + response)
        key = tuple(prompt)
        assert key not in responses_by_prompt
        responses_by_prompt[key] = response
    assert len(prompts) == len({tuple(prompt) for prompt in prompts})
    assert set(responses_by_prompt) == {tuple(prompt) for prompt in prompts}
    return [responses_by_prompt[tuple(prompt)] for prompt in prompts]


def save_fixed_responses(path: Path, prompts: list[list[int]], responses: list[list[int]]) -> None:
    payload = {
        "schema_version": 1,
        "datums": [
            {"prompt_token_ids": prompt, "response_token_ids": response}
            for prompt, response in zip(prompts, responses, strict=True)
        ],
    }
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    load_fixed_responses(temporary, prompts)
    temporary.replace(path)
