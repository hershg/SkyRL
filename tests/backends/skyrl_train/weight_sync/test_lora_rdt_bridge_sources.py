from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt.bridge_sources import (
    extract_lora_bridge_sources,
    validate_lora_bridge_source_layout,
)


def _record(**overrides):
    fields = {
        "global_param_name": "decoder.layers.0.mlp.linear_fc1.adapter.linear_in.weight",
        "hf_param_names": ("base_model.model.layers.0.mlp.gate_proj.lora_A.weight",),
        "component": "linear_in",
        "transform": "identity",
        "weight": torch.ones((2, 4), dtype=torch.float32),
        "tensor_parallel_axis": 1,
        "tensor_parallel_rank": 0,
        "tensor_parallel_size": 2,
        "expert_parallel_axis": None,
        "expert_parallel_rank": 0,
        "expert_parallel_size": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_extract_lora_bridge_sources_separates_fp32_storage_from_stable_metadata():
    tensors, sources = extract_lora_bridge_sources([_record()])

    assert tensors.keys() == {sources[0].key}
    assert tensors[sources[0].key].dtype is torch.float32
    assert sources[0].shape == (2, 4)
    assert sources[0].hf_param_names == (
        "base_model.model.layers.0.mlp.gate_proj.lora_A.weight",
    )
    assert validate_lora_bridge_source_layout(sources)[sources[0].key] == sources[0]


def test_extract_lora_bridge_sources_rejects_duplicate_or_non_fp32_sources():
    record = _record()
    with pytest.raises(ValueError, match="duplicate"):
        extract_lora_bridge_sources([record, record])
    with pytest.raises(ValueError, match="float32"):
        extract_lora_bridge_sources(
            [_record(weight=torch.ones((2, 4), dtype=torch.bfloat16))]
        )
