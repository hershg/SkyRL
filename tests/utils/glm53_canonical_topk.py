"""Canonical sparse-index ordering for an explicit diagnostic control."""

import torch


def canonicalize_sparse_indices(indices):
    sentinel = torch.iinfo(indices.dtype).max
    ordered = indices.masked_fill(indices == -1, sentinel).sort(dim=-1).values
    indices.copy_(ordered.masked_fill(ordered == sentinel, -1))


class CanonicalSparseTopK:
    def __init__(self):
        self.original = None
        self.calls = 0
        self.shapes = set()

    def topk(self, logits, starts, ends, indices, *arguments):
        result = self.original(logits, starts, ends, indices, *arguments)
        canonicalize_sparse_indices(indices)
        self.calls += 1
        self.shapes.add(tuple(indices.shape))
        return result

    def install(self):
        from vllm import _custom_ops

        assert self.original is None
        self.original = _custom_ops.top_k_per_row_prefill
        _custom_ops.top_k_per_row_prefill = self.topk

    def restore(self):
        from vllm import _custom_ops

        assert self.original is not None
        _custom_ops.top_k_per_row_prefill = self.original
        self.original = None
        return {"calls": self.calls, "shapes": sorted(self.shapes)}
