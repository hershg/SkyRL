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


def validate_qwen_wrapper_layout(modules):
    expected = {
        "embed_tokens": ("VocabParallelEmbeddingWithLoRA", None),
        "lm_head": ("LogitsProcessorWithLoRA", None),
        "qkv_proj": ("MergedQKVParallelLinearWithLoRA", 3),
        "gate_up_proj": ("MergedColumnParallelLinearWithLoRA", 2),
        "o_proj": ("RowParallelLinearWithLoRA", 1),
        "down_proj": ("RowParallelLinearWithLoRA", 1),
    }
    errors = []
    for module in modules:
        target = module["name"].rsplit(".", 1)[-1]
        actual = module["class"], module["slices"]
        if target not in expected or actual != expected[target]:
            errors.append({"name": module["name"], "actual": actual, "expected": expected.get(target)})
    assert not errors, errors


def build_b_only_candidate(zero, direction):
    assert zero.keys() == direction.keys()
    candidate = {}
    norms = {}
    for name, tensor in zero.items():
        assert tensor.dtype == direction[name].dtype == torch.float32
        if ".lora_A." in name:
            candidate[name] = tensor.clone()
            assert torch.equal(candidate[name], tensor)
        else:
            assert ".lora_B." in name and torch.count_nonzero(tensor) == 0, name
            candidate[name] = direction[name] * 10
            assert torch.isfinite(candidate[name]).all() and torch.count_nonzero(candidate[name]) > 0, name
            norms[name] = {"original": direction[name].norm().item(), "candidate": candidate[name].norm().item()}
    return candidate, norms


def map_exported_tensor(name):
    prefix = "base_model.model."
    assert name.startswith(prefix), name
    path, adapter, suffix = name[len(prefix) :].rsplit(".", 2)
    assert suffix == "weight" and adapter in ("lora_A", "lora_B"), name
    parent, target = path.rsplit(".", 1)
    fused, index = PACKED_MODULES[target]
    return f"{parent}.{fused}", index, adapter[-1]


def check_untargeted_buffers(name, buffers):
    assert name in ("model.embed_tokens", "lm_head"), name
    for buffer in buffers:
        assert torch.isfinite(buffer).all() and torch.count_nonzero(buffer) == 0, name


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
