"""Load immutable token inputs for GLM publication comparisons."""

import hashlib
import json
from pathlib import Path


def load_parity_inputs(path: str, expected_sha256: str, model_revision: str):
    raw = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Parity input SHA-256 mismatch")
    payload = json.loads(raw)
    if payload["model_revision"] != model_revision:
        raise ValueError("Parity input model revision mismatch")
    prompts = payload["prompt_token_ids"]
    responses = payload["response_ids"]
    if not prompts or len(prompts) != len(responses):
        raise ValueError("Parity inputs require paired prompts and responses")
    for prompt, response in zip(prompts, responses, strict=True):
        if not prompt or not response or len(prompt) + len(response) >= 32768:
            raise ValueError("Parity input must leave room for the scoring token")
        if any(type(token) is not int or token < 0 for token in prompt + response):
            raise ValueError("Parity inputs require nonnegative integer token IDs")
    return prompts, responses
