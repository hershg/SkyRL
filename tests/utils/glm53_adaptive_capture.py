"""Retain paired early-layer inputs and archive rows selected by actual differences."""

import hashlib
import time

import torch

from tests.utils.glm53_sparse_capture import (
    SparseCapture,
    compact_attention_inputs,
    save_capture,
)


def count_tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(count_tensor_bytes(item) for item in value.values())
    return 0


def select_changed_rows(before, after, control_rows=64, changed_rows=8):
    token_count = before["attention"]["query"].shape[0]
    assert token_count == after["attention"]["query"].shape[0]
    changed = {}
    selected = [torch.arange(max(0, token_count - control_rows), token_count)]
    for kind, name in (("attention", "query"), ("attention", "output"), ("indexer", "logits")):
        left, right = before[kind][name], after[kind][name]
        assert left.shape == right.shape
        rows = (left != right).reshape(token_count, -1).any(-1).nonzero().flatten()
        changed[name] = rows
        selected.append(rows[:changed_rows])
    return torch.cat(selected).unique(sorted=True), changed


def slice_attention_capture(capture, rows):
    arguments = {
        **capture["scalars"],
        **{key: capture[key] for key in ("query", "kv_cache", "block_tables", "seq_lens")},
    }
    sliced = compact_attention_inputs(arguments, capture["output"], rows)
    sliced["physical_ids"] = capture["physical_ids"][sliced["physical_ids"].long()]
    sliced["original_indices"] = capture["original_indices"][rows].clone()
    return sliced


class AdaptiveSparseCapture(SparseCapture):
    def __init__(self, directory, token_count, max_records=3, max_retained_bytes=4 * 1024**3):
        super().__init__(directory, token_count, max_records)
        self.max_retained_bytes = max_retained_bytes
        self.pass_index = 0
        self.baseline = {}
        self.pending_indexers = {}
        self.peak_retained_bytes = 0
        self.saved_bytes = 0
        self.reports = []
        self.passes = []

    def install(self):
        assert self.pass_index in (0, 1)
        self.counts = {"attention": 0, "indexer": 0}
        self.native_counts = {"attention": 0, "indexer": 0}
        self.capture_seconds = 0.0
        super().install()

    def check_retained_bytes(self, current=None):
        retained = count_tensor_bytes(self.baseline) + count_tensor_bytes(self.pending_indexers)
        retained += count_tensor_bytes(current)
        self.peak_retained_bytes = max(self.peak_retained_bytes, retained)
        assert retained <= self.max_retained_bytes

    def topk(self, logits, starts, ends, indices, *arguments):
        output = self.original_topk(logits, starts, ends, indices, *arguments)
        if logits.shape[0] != self.token_count:
            return output
        self.native_counts["indexer"] += 1
        index = self.counts["indexer"]
        if index >= self.max_records:
            return output
        started = time.monotonic()
        selected = logits.to("cpu", copy=True)
        cpu_starts, cpu_ends = starts.to("cpu", copy=True), ends.to("cpu", copy=True)
        columns = torch.arange(logits.shape[1])
        valid = (columns >= cpu_starts[:, None]) & (columns < cpu_ends[:, None])
        selected.masked_fill_(~valid, float("-inf"))
        assert torch.isfinite(selected[valid]).all()
        self.pending_indexers[index] = {
            "rows": torch.arange(self.token_count),
            "logits": selected,
            "starts": cpu_starts,
            "ends": cpu_ends,
            "indices": indices.to("cpu", copy=True),
        }
        self.counts["indexer"] += 1
        self.check_retained_bytes()
        self.capture_seconds += time.monotonic() - started
        return output

    def decode(self, **arguments):
        output = self.original_decode(**arguments)
        query = arguments["query"]
        if query.shape[0] != self.token_count:
            return output
        self.native_counts["attention"] += 1
        index = self.counts["attention"]
        if index >= self.max_records:
            return output
        started = time.monotonic()
        assert isinstance(output, torch.Tensor)
        rows = torch.arange(self.token_count, device=query.device)
        capture = compact_attention_inputs(arguments, output, rows)
        for key in ("query", "kv_cache", "output"):
            assert torch.isfinite(capture[key]).all()
        current = {"attention": capture, "indexer": self.pending_indexers.pop(index)}
        if self.pass_index == 0:
            self.baseline[index] = current
            self.check_retained_bytes()
        else:
            previous = self.baseline.pop(index)
            self.check_retained_bytes({"before": previous, "after": current})
            self.save_pair(index, previous, current)
        self.counts["attention"] += 1
        self.capture_seconds += time.monotonic() - started
        return output

    def save_pair(self, index, before, after):
        assert torch.equal(before["indexer"]["starts"], after["indexer"]["starts"])
        assert torch.equal(before["indexer"]["ends"], after["indexer"]["ends"])
        assert before["attention"]["scalars"] == after["attention"]["scalars"]
        rows, changed = select_changed_rows(before, after)
        assert rows.numel() <= 88
        selection = {"rows": rows, "changed_rows": changed, "original_token_count": self.token_count}
        self.saved_bytes += save_capture(self.directory / f"selection_{index:03d}.pt", selection)
        for repeat, capture in enumerate((before, after)):
            directory = self.directory / f"base_{repeat}"
            self.saved_bytes += save_capture(
                directory / f"attention_{index:03d}.pt", slice_attention_capture(capture["attention"], rows)
            )
            self.saved_bytes += save_capture(
                directory / f"indexer_{index:03d}.pt",
                {key: value[rows].clone() for key, value in capture["indexer"].items()},
            )
        assert self.saved_bytes <= 160 * 1024**2
        self.reports.append(
            {
                "layer": index,
                "selected_rows": rows.tolist(),
                "changed_row_counts": {key: value.numel() for key, value in changed.items()},
                "first_changed_rows": {key: value[:8].tolist() for key, value in changed.items()},
            }
        )

    def restore(self):
        super().restore()
        assert self.counts == {"attention": self.max_records, "indexer": self.max_records}
        assert not self.pending_indexers
        self.passes.append(
            {"captured": dict(self.counts), "native": dict(self.native_counts), "capture_seconds": self.capture_seconds}
        )
        self.pass_index += 1
        pending = self.pass_index == 1
        if pending:
            assert len(self.baseline) == self.max_records
        else:
            assert self.pass_index == 2 and not self.baseline
        files = []
        for path in sorted(self.directory.rglob("*.pt")):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files.append(
                {"name": str(path.relative_to(self.directory)), "bytes": path.stat().st_size, "sha256": digest}
            )
        return {
            "directory": str(self.directory),
            "pending": pending,
            "passes": self.passes,
            "reports": self.reports,
            "peak_retained_tensor_bytes": self.peak_retained_bytes,
            "saved_bytes": self.saved_bytes,
            "files": files,
        }
