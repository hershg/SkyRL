import hashlib
import json

import pytest

from tests.utils.glm53_parity_inputs import load_parity_inputs


def _write_inputs(tmp_path, prompts, responses, revision="test-revision"):
    path = tmp_path / "inputs.json"
    raw = json.dumps(
        {
            "model_revision": revision,
            "prompt_token_ids": prompts,
            "response_ids": responses,
        }
    ).encode()
    path.write_bytes(raw)
    return str(path), hashlib.sha256(raw).hexdigest()


def test_parity_inputs_preserve_heterogeneous_sequences_and_order(tmp_path):
    prompts, responses = [[4, 5], [9]], [[7], [11, 12, 13]]
    path, digest = _write_inputs(tmp_path, prompts, responses)
    assert load_parity_inputs(path, digest, "test-revision") == (prompts, responses)


def test_parity_inputs_reject_replaced_tokens(tmp_path):
    path, digest = _write_inputs(tmp_path, [[4]], [[7]])
    _write_inputs(tmp_path, [[4]], [[8]])
    with pytest.raises(ValueError, match="SHA-256"):
        load_parity_inputs(path, digest, "test-revision")


def test_parity_inputs_reject_wrong_model_revision(tmp_path):
    path, digest = _write_inputs(tmp_path, [[4]], [[7]])
    with pytest.raises(ValueError, match="revision"):
        load_parity_inputs(path, digest, "different-revision")


@pytest.mark.parametrize(
    "prompts,responses",
    [
        ([], []),
        ([[4]], []),
        ([[]], [[7]]),
        ([[4]], [[]]),
        ([[4]], [[True]]),
        ([[4]], [[-1]]),
        ([[4]], [[1.5]]),
        ([[4] * 32767], [[7]]),
    ],
)
def test_parity_inputs_reject_incomplete_or_invalid_scoring_work(
    tmp_path, prompts, responses
):
    path, digest = _write_inputs(tmp_path, prompts, responses)
    with pytest.raises(ValueError):
        load_parity_inputs(path, digest, "test-revision")
