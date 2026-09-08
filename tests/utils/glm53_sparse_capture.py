"""Capture sparse MLA inputs without changing the kernel's returned tensors."""

import hashlib
import os
import tempfile
from pathlib import Path

import torch


def compact_attention_inputs(arguments, output, rows):
    indices = arguments["block_tables"][rows].clone()
    valid = indices >= 0
    physical_ids = indices[valid].unique(sorted=True)
    assert physical_ids.numel() > 0
    cache = arguments["kv_cache"]
    assert cache.ndim == 4 and cache.shape[1] == 1
    flat_cache = cache.reshape(-1, cache.shape[-1])
    assert physical_ids.max() < flat_cache.shape[0]
    compact_indices = torch.searchsorted(physical_ids, indices.clamp_min(0))
    compact_indices[~valid] = -1
    block_size = cache.shape[2]
    padded_length = ((physical_ids.numel() + block_size - 1) // block_size) * block_size
    compact_cache = cache.new_zeros((padded_length, cache.shape[-1]))
    compact_cache[: physical_ids.numel()] = flat_cache[physical_ids.long()]
    scalars = {
        key: arguments[key]
        for key in (
            "qk_nope_head_dim",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "max_seq_len",
            "bmm1_scale",
            "bmm2_scale",
            "sparse_mla_top_k",
            "return_lse",
        )
    }
    assert scalars["return_lse"] is False
    return {
        "rows": rows.cpu(),
        "original_query_shape": tuple(arguments["query"].shape),
        "physical_ids": physical_ids.cpu(),
        "original_indices": indices.cpu(),
        "query": arguments["query"][rows].cpu(),
        "kv_cache": compact_cache.reshape(-1, 1, block_size, cache.shape[-1]).cpu(),
        "block_tables": compact_indices.to(indices.dtype).cpu(),
        "seq_lens": arguments["seq_lens"][rows].cpu(),
        "output": output[rows].cpu(),
        "scalars": scalars,
    }


class SparseCapture:
    def __init__(self, directory, token_count, max_records=None):
        self.directory = Path(directory)
        self.token_count = token_count
        assert max_records is None or max_records > 0
        self.max_records = max_records
        self.counts = {"attention": 0, "indexer": 0}
        self.original_decode = None
        self.original_topk = None

    def save(self, kind, tensors):
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{kind}_{self.counts[kind]:03d}.pt"
        assert not destination.exists()
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
            torch.save(tensors, handle)
            temporary = handle.name
        os.replace(temporary, destination)
        self.counts[kind] += 1

    def decode(self, **arguments):
        output = self.original_decode(**arguments)
        query = arguments["query"]
        if query.shape[0] == self.token_count and (
            self.max_records is None or self.counts["attention"] < self.max_records
        ):
            assert isinstance(output, torch.Tensor)
            rows = torch.arange(max(0, self.token_count - 64), self.token_count, device=query.device)
            self.save("attention", compact_attention_inputs(arguments, output, rows))
        return output

    def topk(self, logits, starts, ends, indices, *arguments):
        output = self.original_topk(logits, starts, ends, indices, *arguments)
        if logits.shape[0] == self.token_count and (
            self.max_records is None or self.counts["indexer"] < self.max_records
        ):
            rows = torch.arange(max(0, self.token_count - 64), self.token_count, device=logits.device)
            selected = logits[rows].clone()
            columns = torch.arange(logits.shape[1], device=logits.device)
            valid = (columns >= starts[rows, None]) & (columns < ends[rows, None])
            selected.masked_fill_(~valid, float("-inf"))
            assert torch.isfinite(selected[valid]).all()
            self.save(
                "indexer",
                {
                    "rows": rows.cpu(),
                    "logits": selected.cpu(),
                    "starts": starts[rows].cpu(),
                    "ends": ends[rows].cpu(),
                    "indices": indices[rows].cpu(),
                },
            )
        return output

    def install(self):
        from flashinfer import decode
        from vllm import _custom_ops

        assert self.original_decode is None and self.original_topk is None
        self.original_decode = decode.trtllm_batch_decode_with_kv_cache_mla
        self.original_topk = _custom_ops.top_k_per_row_prefill
        decode.trtllm_batch_decode_with_kv_cache_mla = self.decode
        _custom_ops.top_k_per_row_prefill = self.topk

    def restore(self):
        from flashinfer import decode
        from vllm import _custom_ops

        assert self.original_decode is not None and self.original_topk is not None
        decode.trtllm_batch_decode_with_kv_cache_mla = self.original_decode
        _custom_ops.top_k_per_row_prefill = self.original_topk
        self.original_decode = None
        self.original_topk = None
        files = []
        for path in sorted(self.directory.glob("*.pt")):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
        return {"directory": str(self.directory), "counts": self.counts, "files": files}
