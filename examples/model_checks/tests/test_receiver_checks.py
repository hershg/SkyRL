from copy import deepcopy

import pytest

from examples.model_checks.receiver_checks import (
    fingerprint_receivers,
    validate_receivers,
)


@pytest.fixture
def receipts():
    return [
        {
            "engine_id": engine,
            "tp_rank": rank,
            "adapter_id": 1,
            "context": 32768,
            "model_dtype": "torch.bfloat16",
            "buffers": {"adapter": {"dtype": "torch.bfloat16", "sha256": f"{engine}:{rank}"}},
            "kv_tensors": [{"dtype": "torch.bfloat16"}],
        }
        for engine in ("engine-a", "engine-b")
        for rank in range(8)
    ]


def test_two_engine_fingerprints_preserve_every_rank_and_detect_one_changed_buffer(receipts):
    engines = ["engine-a", "engine-b"]
    before = fingerprint_receivers(receipts, engines, 8)
    assert sum(len(ranks) for ranks in before.values()) == 16
    assert fingerprint_receivers(list(reversed(receipts)), engines, 8) == before
    updated = deepcopy(receipts)
    updated[0]["buffers"]["adapter"]["sha256"] = "changed"
    after = fingerprint_receivers(updated, engines, 8)
    assert before["engine-a"] != after["engine-a"]
    assert before["engine-b"] == after["engine-b"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown_engine", "adapter", "kv_dtype", "context"])
def test_receiver_audit_rejects_incomplete_or_inconsistent_engine_evidence(receipts, failure):
    if failure == "missing":
        receipts.pop()
    elif failure == "duplicate":
        receipts[-1] = deepcopy(receipts[0])
    elif failure == "unknown_engine":
        receipts[-1]["engine_id"] = "unadmitted"
    elif failure == "adapter":
        receipts[-1]["adapter_id"] = 2
    elif failure == "kv_dtype":
        receipts[-1]["kv_tensors"][0]["dtype"] = "torch.float32"
    elif failure == "context":
        receipts[-1]["context"] = 8192
    with pytest.raises(ValueError):
        validate_receivers(receipts, ["engine-a", "engine-b"], 8)


@pytest.mark.parametrize("engines", [["engine-a", "engine-a", "engine-b"], ["engine-a", ""], []])
def test_receiver_audit_rejects_invalid_expected_engine_contract(receipts, engines):
    with pytest.raises(ValueError, match="nonempty and unique"):
        validate_receivers([item for item in receipts if item["engine_id"] in engines], engines, 8)


def test_receiver_adapter_ids_are_local_to_each_engine(receipts):
    for item in receipts:
        if item["engine_id"] == "engine-b":
            item["adapter_id"] = 2
    assert len(validate_receivers(receipts, ["engine-a", "engine-b"], 8)) == 16
