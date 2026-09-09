"""Diagnostic-only Qwen TP1 mapping; deliberately independent of vLLM's loader."""

import torch

PACKED_MODULES = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "o_proj": ("o_proj", 0),
    "down_proj": ("down_proj", 0),
}


def map_exported_tensor(name):
    prefix = "base_model.model."
    assert name.startswith(prefix), name
    path, adapter, suffix = name[len(prefix) :].rsplit(".", 2)
    assert suffix == "weight" and adapter in ("lora_A", "lora_B"), name
    parent, target = path.rsplit(".", 1)
    fused, index = PACKED_MODULES[target]
    return f"{parent}.{fused}", index, adapter[-1]


def compare_active_tensors(exported, loaded, rank, alpha):
    expected_keys = set()
    mismatches = []
    elements = 0
    for name, tensor in exported.items():
        key = map_exported_tensor(name)
        assert key not in expected_keys, key
        expected_keys.add(key)
        actual = loaded[key]
        assert actual.device.type == "cpu" and tensor.ndim == actual.ndim == 2, name
        assert torch.isfinite(actual).all() and torch.isfinite(tensor).all(), name
        assert tensor.shape[0 if key[2] == "A" else 1] == rank, name
        expected = tensor.to(actual.dtype)
        if key[2] == "B":
            expected = expected * (alpha / rank)
        assert all(want <= have for want, have in zip(expected.shape, actual.shape)), name
        padded = torch.zeros_like(actual)
        padded[: expected.shape[0], : expected.shape[1]] = expected
        if not torch.equal(actual, padded):
            mismatches.append({"tensor": name, "max_abs": (actual.float() - padded.float()).abs().max().item()})
        elements += tensor.numel()
    assert expected_keys == loaded.keys(), "Missing or unexpected active target tensors"
    return {"passed": not mismatches, "tensors": len(exported), "elements": elements, "mismatches": mismatches}
