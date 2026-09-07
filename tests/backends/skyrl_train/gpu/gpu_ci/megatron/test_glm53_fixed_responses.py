import json

import pytest

from tests.backends.skyrl_train.gpu.gpu_ci.megatron.glm53_fixed_responses import (
    load_fixed_responses,
    save_fixed_responses,
)


def test_replay_matches_prompt_tokens_not_artifact_order(tmp_path):
    path = tmp_path / "responses.json"
    save_fixed_responses(path, [[11, 12], [21]], [[31], [41, 42]])
    assert load_fixed_responses(path, [[21], [11, 12]]) == [[41, 42], [31]]


@pytest.mark.parametrize("prompts", [[[11]], [[11, 12], [22]], [[11, 12], [11, 12]]])
def test_replay_rejects_changed_or_missing_prompts(tmp_path, prompts):
    path = tmp_path / "responses.json"
    save_fixed_responses(path, [[11, 12], [21]], [[31], [41, 42]])
    with pytest.raises(AssertionError):
        load_fixed_responses(path, prompts)


@pytest.mark.parametrize("response", [[], [-1], [True], [1.5]])
def test_replay_rejects_invalid_response_tokens(tmp_path, response):
    path = tmp_path / "responses.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datums": [
                    {"prompt_token_ids": [11], "response_token_ids": response},
                ],
            }
        )
    )
    with pytest.raises(AssertionError):
        load_fixed_responses(path, [[11]])


def test_replay_rejects_duplicate_prompt_artifacts(tmp_path):
    path = tmp_path / "responses.json"
    datum = {"prompt_token_ids": [11], "response_token_ids": [31]}
    path.write_text(json.dumps({"schema_version": 1, "datums": [datum, datum]}))
    with pytest.raises(AssertionError):
        load_fixed_responses(path, [[11]])
