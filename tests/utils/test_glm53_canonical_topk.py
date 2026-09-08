import sys
from types import SimpleNamespace

import pytest
import torch

from tests.utils.glm53_canonical_topk import (
    CanonicalSparseTopK,
    canonicalize_sparse_indices,
)
from tests.utils.glm53_sparse_capture import SparseCapture


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_ordering_preserves_selected_tokens_and_padding_suffix(dtype):
    indices = torch.tensor([[9, 2, -1, 4], [-1, -1, -1, -1], [8, 5, 1, 3]], dtype=dtype)
    canonicalize_sparse_indices(indices)
    assert indices.tolist() == [[2, 4, 9, -1], [-1, -1, -1, -1], [1, 3, 5, 8]]


def test_ordering_updates_noncontiguous_output_without_changing_neighbors():
    storage = torch.tensor([[9, 99, 2, 99, -1, 99, 4, 99]], dtype=torch.int32)
    canonicalize_sparse_indices(storage[:, ::2])
    assert storage.tolist() == [[2, 99, 4, 99, 9, 99, -1, 99]]


def test_equal_selected_sets_have_identical_order():
    first = torch.tensor([[8, 4, 9, 3]])
    second = torch.tensor([[9, 3, 4, 8]])
    canonicalize_sparse_indices(first)
    canonicalize_sparse_indices(second)
    assert torch.equal(first, second)


def test_control_restores_native_operation_and_preserves_its_return(monkeypatch):
    marker = object()

    def native(logits, starts, ends, indices, *arguments):
        indices.copy_(torch.tensor([[7, 2, -1]], dtype=indices.dtype))
        return marker

    operations = SimpleNamespace(top_k_per_row_prefill=native)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=operations))
    control = CanonicalSparseTopK()
    control.install()
    indices = torch.empty((1, 3), dtype=torch.int32)
    assert operations.top_k_per_row_prefill(None, None, None, indices) is marker
    assert indices.tolist() == [[2, 7, -1]]
    receipt = control.restore()
    assert operations.top_k_per_row_prefill is native
    assert receipt == {"calls": 1, "shapes": [(1, 3)]}


def test_capture_records_canonical_indices_and_restores_the_outer_control(tmp_path, monkeypatch):
    marker = object()

    def native(logits, starts, ends, indices, *arguments):
        indices.copy_(torch.tensor([[2, 0, -1]], dtype=indices.dtype))
        return marker

    def decode(**arguments):
        return arguments["query"]

    operations = SimpleNamespace(top_k_per_row_prefill=native)
    decoder = SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=decode)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(_custom_ops=operations))
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(decode=decoder))
    control = CanonicalSparseTopK()
    control.install()
    capture = SparseCapture(tmp_path, 1)
    logits = torch.tensor([[3.0, 1.0, 4.0]])
    starts, ends = torch.tensor([0]), torch.tensor([3])
    indices = torch.empty((1, 3), dtype=torch.int32)
    try:
        capture.install()
        try:
            assert operations.top_k_per_row_prefill(logits, starts, ends, indices) is marker
        finally:
            receipt = capture.restore()
        assert receipt["counts"] == {"attention": 0, "indexer": 1}
        saved = torch.load(tmp_path / "indexer_000.pt", weights_only=True)
        assert saved["indices"].tolist() == [[0, 2, -1]]
        assert operations.top_k_per_row_prefill(logits, starts, ends, indices) is marker
        assert indices.tolist() == [[0, 2, -1]]
        assert decoder.trtllm_batch_decode_with_kv_cache_mla is decode
    finally:
        control_receipt = control.restore()
    assert operations.top_k_per_row_prefill is native
    assert control_receipt == {"calls": 2, "shapes": [(1, 3)]}
