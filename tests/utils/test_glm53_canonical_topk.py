import sys
from types import SimpleNamespace

import pytest
import torch

from tests.utils.glm53_canonical_topk import (
    CanonicalSparseTopK,
    canonicalize_sparse_indices,
)


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
