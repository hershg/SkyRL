"""Opt-in named gradient norms for Megatron optimizers."""

import contextlib
import fnmatch
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

NAMED_GRADIENT_METRIC_PREFIX = "skyrl.ai/named_grad_norm/"


def _strip_module_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


def _iter_group_pairs(model_groups: Sequence, main_groups: Sequence) -> Iterator[tuple[Any, Any]]:
    for model_group, main_group in zip(model_groups, main_groups, strict=True):
        yield from zip(model_group, main_group, strict=True)


def _iter_model_main_params(optimizer) -> Iterator[tuple[Any, Any]]:
    """Yield model parameters and their prepared optimizer parameters."""
    if hasattr(optimizer, "model_float16_groups"):
        if optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            float_main_groups = optimizer.shard_float16_groups
        else:
            float_main_groups = optimizer.shard_fp32_from_float16_groups
        yield from _iter_group_pairs(optimizer.model_float16_groups, float_main_groups)
        yield from _iter_group_pairs(optimizer.model_fp32_groups, optimizer.shard_fp32_groups)
        return

    if hasattr(optimizer, "float16_groups"):
        yield from _iter_group_pairs(optimizer.float16_groups, optimizer.fp32_from_float16_groups)
        yield from _iter_group_pairs(optimizer.fp32_from_fp32_groups, optimizer.fp32_from_fp32_groups)
        return

    for param in optimizer.get_parameters():
        yield param, param


class NamedGradientRecorder:
    """Collect configured norms from prepared gradients before clipping."""

    def __init__(self, optimizer, model_chunks: Sequence, selectors: Mapping[str, str]):
        self._optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        self._labels = tuple(selectors)

        selected_model_params = {label: set() for label in self._labels}
        overlap = False
        for chunk in model_chunks:
            for raw_name, param in chunk.named_parameters():
                name = _strip_module_prefix(raw_name)
                matches = [label for label, pattern in selectors.items() if fnmatch.fnmatchcase(name, pattern)]
                overlap |= len(matches) > 1
                for label in matches:
                    selected_model_params[label].add(id(param))

        self._validate_global_selection(selected_model_params, overlap, selectors)
        self._params = []
        for sub_optimizer in self._optimizers:
            params_by_label = {label: [] for label in self._labels}
            for model_param, main_param in _iter_model_main_params(sub_optimizer):
                for label, selected in selected_model_params.items():
                    if id(model_param) in selected:
                        params_by_label[label].append(main_param)
            self._params.append(params_by_label)

    @staticmethod
    def _validate_global_selection(selected, overlap, selectors) -> None:
        flags = [int(bool(selected[label])) for label in selectors]
        flags.append(int(overlap))
        if torch.distributed.is_initialized():
            values = torch.tensor(flags, dtype=torch.int, device=torch.cuda.current_device())
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
            flags = values.cpu().tolist()
        missing = [label for label, found in zip(selectors, flags[:-1], strict=True) if not found]
        if missing:
            raise ValueError(f"Named-gradient selectors matched no parameters: {missing}")
        if flags[-1]:
            raise ValueError("Named-gradient selectors must not overlap")

    @torch.no_grad()
    def _collect(self) -> dict[str, float]:
        metrics = {}
        for label in self._labels:
            sub_norms = []
            for sub_optimizer, params_by_label in zip(self._optimizers, self._params, strict=True):
                grads = sub_optimizer._filter_grads_for_norm(params_by_label[label])
                norm = get_grad_norm_fp32(
                    grads,
                    grad_stats_parallel_group=sub_optimizer.get_grad_stats_parallel_group(),
                )
                sub_norms.append(float(norm))
            value = math.sqrt(sum(norm**2 for norm in sub_norms))
            if not math.isfinite(value):
                raise FloatingPointError(f"Named gradient norm {label!r} is non-finite: {value}")
            metrics[f"{NAMED_GRADIENT_METRIC_PREFIX}{label}"] = value
        return metrics

    @contextlib.contextmanager
    def capture(self, optimizer) -> Iterator[dict[str, float]]:
        """Capture once after ``prepare_grads`` and before clipping or mutation."""
        metrics = {}
        original_prepare_grads: Callable = optimizer.prepare_grads

        def prepare_grads():
            found_inf = original_prepare_grads()
            if not metrics:
                metrics.update(self._collect())
            return found_inf

        optimizer.prepare_grads = prepare_grads
        try:
            yield metrics
        finally:
            optimizer.prepare_grads = original_prepare_grads
