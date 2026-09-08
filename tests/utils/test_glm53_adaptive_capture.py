import sys
from types import SimpleNamespace

import pytest
import torch

from tests.utils.glm53_adaptive_capture import (
    AdaptiveSparseCapture,
    count_tensor_bytes,
    select_changed_rows,
    slice_attention_capture,
)
from tests.utils.glm53_sparse_capture import compact_attention_inputs
from tests.utils.test_glm53_sparse_capture import make_arguments


def test_row_selection_includes_early_drift_and_bounds_dense_differences():
    before = {
        "attention": {"query": torch.zeros(128, 2), "output": torch.zeros(128, 2)},
        "indexer": {"logits": torch.zeros(128, 3)},
    }
    after = {kind: {name: value.clone() for name, value in values.items()} for kind, values in before.items()}
    after["attention"]["query"][3] = 1
    after["attention"]["output"][12:30] = 2
    after["indexer"]["logits"][40:60] = 3
    rows, changed = select_changed_rows(before, after, control_rows=4, changed_rows=2)
    assert rows.tolist() == [3, 12, 13, 40, 41, 124, 125, 126, 127]
    assert changed["query"].tolist() == [3]
    assert changed["output"].tolist() == list(range(12, 30))
    assert changed["logits"].tolist() == list(range(40, 60))


def test_second_compaction_preserves_original_physical_ids_and_replay_values():
    arguments = make_arguments()
    full = compact_attention_inputs(arguments, arguments["query"], torch.arange(3))
    selected = slice_attention_capture(full, torch.tensor([2]))
    assert selected["physical_ids"].tolist() == [0, 5, 11]
    assert selected["original_indices"].tolist() == [[[0, 11, 5]]]
    actual = selected["kv_cache"].reshape(-1, 4)[selected["block_tables"].long()]
    expected = arguments["kv_cache"].reshape(-1, 4)[arguments["block_tables"][[2]].long()]
    assert torch.equal(actual, expected)
    assert selected["original_query_shape"] == (3, 1, 2, 4)


def test_paired_capture_keeps_native_calls_and_releases_host_snapshots(tmp_path, monkeypatch):
    arguments = make_arguments()
    logits = torch.ones(3, 3)
    starts, ends = torch.zeros(3, dtype=torch.int32), torch.full((3,), 3, dtype=torch.int32)
    indices = torch.empty(3, 3, dtype=torch.int32)
    output = arguments["query"] + 1
    calls = {"attention": 0, "indexer": 0}

    def decode(**kwargs):
        calls["attention"] += 1
        return output

    def topk(logits, starts, ends, indices):
        calls["indexer"] += 1
        indices.copy_(torch.arange(3).expand(3, 3))
        return "native result"

    decoder = SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=decode)
    operations = SimpleNamespace(top_k_per_row_prefill=topk)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(decode=decoder))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=operations))
    capture = AdaptiveSparseCapture(tmp_path, 3)
    for repeat in range(2):
        capture.install()
        for _ in range(4):
            assert operations.top_k_per_row_prefill(logits, starts, ends, indices) == "native result"
            assert decoder.trtllm_batch_decode_with_kv_cache_mla(**arguments) is output
        receipt = capture.restore()
        assert decoder.trtllm_batch_decode_with_kv_cache_mla is decode
        assert operations.top_k_per_row_prefill is topk
        assert receipt["pending"] == (repeat == 0)
        if repeat == 0:
            assert not receipt["files"]
            assert len(capture.baseline) == 3
            output[0] += 1
            arguments["query"][1] += 2
            logits[2] += 3
    assert calls == {"attention": 8, "indexer": 8}
    assert not capture.baseline and not capture.pending_indexers
    assert len(receipt["files"]) == 15
    assert receipt["saved_bytes"] == sum(item["bytes"] for item in receipt["files"])
    assert all(p["native"] == {"attention": 4, "indexer": 4} for p in receipt["passes"])
    assert all(p["captured"] == {"attention": 3, "indexer": 3} for p in receipt["passes"])
    assert all(r["first_changed_rows"] == {"query": [1], "output": [0], "logits": [2]} for r in receipt["reports"])
    with pytest.raises(AssertionError):
        capture.install()


def test_host_retention_limit_fails_before_an_unbounded_second_snapshot(tmp_path):
    capture = AdaptiveSparseCapture(tmp_path, 3, max_retained_bytes=16)
    capture.baseline = {0: {"query": torch.ones(4)}}
    assert count_tensor_bytes(capture.baseline) == 16
    capture.check_retained_bytes()
    with pytest.raises(AssertionError):
        capture.check_retained_bytes({"query": torch.ones(1)})


def test_row_selection_rejects_incompatible_pair_geometry():
    before = {"attention": {"query": torch.ones(3, 2)}}
    after = {"attention": {"query": torch.ones(2, 2)}}
    with pytest.raises(AssertionError):
        select_changed_rows(before, after)
