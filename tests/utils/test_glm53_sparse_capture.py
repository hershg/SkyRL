import sys
from types import SimpleNamespace

import pytest
import torch

from tests.utils.glm53_sparse_capture import SparseCapture, compact_attention_inputs


def make_arguments():
    return {
        "query": torch.arange(24, dtype=torch.float32).reshape(3, 1, 2, 4),
        "kv_cache": torch.arange(48, dtype=torch.float32).reshape(3, 1, 4, 4),
        "block_tables": torch.tensor([[[9, 2, -1]], [[5, 2, 9]], [[0, 11, 5]]], dtype=torch.int32),
        "seq_lens": torch.tensor([2, 3, 3], dtype=torch.int32),
        "qk_nope_head_dim": 2,
        "kv_lora_rank": 2,
        "qk_rope_head_dim": 2,
        "max_seq_len": 3,
        "bmm1_scale": 0.5,
        "bmm2_scale": 1.0,
        "sparse_mla_top_k": 3,
        "return_lse": False,
    }


def test_compaction_preserves_selected_keys_order_padding_and_inputs():
    arguments = make_arguments()
    originals = {key: value.clone() for key, value in arguments.items() if isinstance(value, torch.Tensor)}
    output = arguments["query"] + 100
    snapshot = compact_attention_inputs(arguments, output, torch.tensor([0, 2]))
    assert snapshot["original_query_shape"] == (3, 1, 2, 4)
    assert torch.equal(snapshot["output"], output[[0, 2]])
    assert snapshot["kv_cache"].shape == (2, 1, 4, 4)
    valid = snapshot["block_tables"] >= 0
    actual = snapshot["kv_cache"].reshape(-1, 4)[snapshot["block_tables"][valid].long()]
    expected = arguments["kv_cache"].reshape(-1, 4)[arguments["block_tables"][[0, 2]][valid].long()]
    assert torch.equal(actual, expected)
    assert torch.equal(snapshot["block_tables"] < 0, snapshot["original_indices"] < 0)
    for key, value in originals.items():
        assert torch.equal(arguments[key], value)


def test_capture_returns_original_output_and_ignores_unselected_length(tmp_path):
    arguments = make_arguments()
    expected = arguments["query"] + 1
    capture = SparseCapture(tmp_path, 3)
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs["query"].shape[0])
        return expected

    capture.original_decode = kernel
    assert capture.decode(**arguments) is expected
    assert capture.counts == {"attention": 1, "indexer": 0}
    saved = torch.load(tmp_path / "attention_000.pt", weights_only=True)
    assert torch.equal(saved["output"], expected)
    arguments["query"] = arguments["query"][:1]
    assert capture.decode(**arguments) is expected
    assert capture.counts["attention"] == 1
    assert calls == [3, 1]
    assert len(list(tmp_path.iterdir())) == 1


def test_indexer_capture_masks_uninitialized_regions_without_mutating_logits(tmp_path):
    capture = SparseCapture(tmp_path, 2)
    logits = torch.tensor([[1.0, 2.0, float("nan"), float("nan")], [float("nan"), 4.0, 3.0, float("nan")]])
    starts, ends = torch.tensor([0, 1]), torch.tensor([2, 3])
    indices = torch.full((2, 2), -1, dtype=torch.int32)

    def kernel(logits, starts, ends, indices, *args):
        indices.copy_(torch.tensor([[1, 0], [0, 1]], dtype=torch.int32))
        return "original result"

    capture.original_topk = kernel
    assert capture.topk(logits, starts, ends, indices) == "original result"
    saved = torch.load(tmp_path / "indexer_000.pt", weights_only=True)
    assert saved["logits"].tolist() == [
        [1, 2, float("-inf"), float("-inf")],
        [float("-inf"), 4, 3, float("-inf")],
    ]
    assert torch.isnan(logits[0, 2:]).all()
    assert torch.equal(saved["indices"], indices)


def test_restore_recovers_original_kernels_after_capture_error(tmp_path, monkeypatch):
    def failing_decode(**arguments):
        raise RuntimeError("kernel failure")

    def original_topk(*arguments):
        return None

    decode = SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=failing_decode)
    operations = SimpleNamespace(top_k_per_row_prefill=original_topk)
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(decode=decode))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=operations))
    capture = SparseCapture(tmp_path, 3)
    capture.install()
    try:
        with pytest.raises(RuntimeError, match="kernel failure"):
            decode.trtllm_batch_decode_with_kv_cache_mla(**make_arguments())
    finally:
        capture.restore()
    assert decode.trtllm_batch_decode_with_kv_cache_mla is failing_decode
    assert operations.top_k_per_row_prefill is original_topk
    assert not list(tmp_path.iterdir())


def test_compaction_rejects_out_of_range_cache_selection():
    arguments = make_arguments()
    arguments["block_tables"][0, 0, 0] = 12
    with pytest.raises(AssertionError):
        compact_attention_inputs(arguments, arguments["query"], torch.tensor([0]))


def test_capture_limit_bounds_files_without_skipping_kernel_execution(tmp_path):
    capture = SparseCapture(tmp_path, 3, max_records=2)
    arguments = make_arguments()
    expected = arguments["query"] + 1
    calls = {"attention": 0, "indexer": 0}

    def decode(**arguments):
        calls["attention"] += 1
        return expected

    def topk(logits, starts, ends, indices):
        calls["indexer"] += 1
        indices.copy_(torch.tensor([[1, 0]] * 3, dtype=indices.dtype))
        return "native"

    capture.original_decode = decode
    capture.original_topk = topk
    logits = torch.ones(3, 2)
    starts, ends = torch.zeros(3, dtype=torch.int32), torch.full((3,), 2)
    indices = torch.empty((3, 2), dtype=torch.int32)
    for _ in range(4):
        assert capture.decode(**arguments) is expected
        assert capture.topk(logits, starts, ends, indices) == "native"
        assert indices.tolist() == [[1, 0]] * 3
    assert capture.counts == {"attention": 2, "indexer": 2}
    assert calls == {"attention": 4, "indexer": 4}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "attention_000.pt",
        "attention_001.pt",
        "indexer_000.pt",
        "indexer_001.pt",
    ]
    assert torch.equal(torch.load(tmp_path / "attention_001.pt", weights_only=True)["output"], expected)
