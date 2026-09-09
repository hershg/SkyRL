import pytest
import torch

from examples.model_checks.active_lora_audit import (
    check_qwen_export_scope,
    check_untargeted_buffers,
    compare_active_tensors,
    map_exported_tensor,
    validate_qwen_wrapper_layout,
)


@pytest.mark.parametrize("layers", [28, 36])
def test_export_scope_uses_exact_model_geometry(layers):
    geometry = dict(
        num_hidden_layers=layers,
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=3,
        intermediate_size=8,
    )
    exported = {}
    for layer in range(layers):
        for target, width_in, width_out in (
            ("self_attn.q_proj", 4, 6),
            ("self_attn.k_proj", 4, 3),
            ("self_attn.v_proj", 4, 3),
            ("self_attn.o_proj", 6, 4),
            ("mlp.gate_proj", 4, 8),
            ("mlp.up_proj", 4, 8),
            ("mlp.down_proj", 8, 4),
        ):
            for label, shape in (("A", (2, width_in)), ("B", (width_out, 2))):
                exported[f"base_model.model.model.layers.{layer}.{target}.lora_{label}.weight"] = torch.zeros(shape)
    check_qwen_export_scope(exported, geometry, 2)
    key = next(iter(exported))
    tensor = exported.pop(key)
    with pytest.raises(AssertionError, match="Missing"):
        check_qwen_export_scope(exported, geometry, 2)
    exported[key] = tensor.transpose(0, 1)
    with pytest.raises(AssertionError):
        check_qwen_export_scope(exported, geometry, 2)


def make_tensors():
    exported, loaded = {}, {}
    for index, target in enumerate(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "o_proj", "down_proj")):
        for label, shape in (("A", (2, 4)), ("B", (4, 2))):
            name = f"base_model.model.model.layers.0.self_attn.{target}.lora_{label}.weight"
            exported[name] = torch.full(shape, index + 0.1)
            loaded[map_exported_tensor(name)] = exported[name].to(torch.bfloat16)
    return exported, loaded


def test_exact_active_mapping_and_scaling():
    exported, loaded = make_tensors()
    for key in loaded:
        if key[2] == "B":
            loaded[key] *= 0.5
    assert compare_active_tensors(exported, loaded, rank=2, alpha=1)["passed"]


def test_swapped_qkv_slices_fail():
    exported, loaded = make_tensors()
    q = ("model.layers.0.self_attn.qkv_proj", 0, "B")
    k = ("model.layers.0.self_attn.qkv_proj", 1, "B")
    loaded[q], loaded[k] = loaded[k], loaded[q]
    assert len(compare_active_tensors(exported, loaded, 2, 2)["mismatches"]) == 2


def test_stale_expected_tensor_fails():
    exported, loaded = make_tensors()
    name = next(iter(exported))
    exported[name] = torch.zeros_like(exported[name])
    assert compare_active_tensors(exported, loaded, 2, 2)["mismatches"][0]["tensor"] == name


def test_nonzero_padding_fails():
    exported, loaded = make_tensors()
    key = next(iter(loaded))
    loaded[key] = torch.cat((loaded[key], torch.ones(1, 4, dtype=torch.bfloat16)))
    assert not compare_active_tensors(exported, loaded, 2, 2)["passed"]


def test_missing_scope_fails():
    exported, loaded = make_tensors()
    loaded["unknown", 0, "A"] = torch.zeros(2, 4)
    with pytest.raises(AssertionError, match="unexpected"):
        compare_active_tensors(exported, loaded, 2, 2)


@pytest.mark.parametrize("name", ["model.embed_tokens", "lm_head"])
def test_untargeted_wrappers_must_stay_zero(name):
    check_untargeted_buffers(name, [torch.zeros(2, 4)])
    with pytest.raises(AssertionError):
        check_untargeted_buffers(name, [torch.ones(2, 4)])


def test_all_qwen028_wrappers_and_slice_counts():
    modules = [
        {"name": name, "class": cls, "slices": slices}
        for name, cls, slices in (
            ("model.embed_tokens", "VocabParallelEmbeddingWithLoRA", None),
            ("lm_head", "LogitsProcessorWithLoRA", None),
            ("model.layers.0.self_attn.qkv_proj", "MergedQKVParallelLinearWithLoRA", 3),
            ("model.layers.0.mlp.gate_up_proj", "MergedColumnParallelLinearWithLoRA", 2),
            ("model.layers.0.self_attn.o_proj", "RowParallelLinearWithLoRA", 1),
            ("model.layers.0.mlp.down_proj", "RowParallelLinearWithLoRA", 1),
        )
    ]
    validate_qwen_wrapper_layout(modules)
    modules[2]["class"] = "QKVParallelLinearWithLoRA"
    modules[3]["slices"] = 3
    with pytest.raises(AssertionError) as error:
        validate_qwen_wrapper_layout(modules)
    assert "qkv_proj" in str(error.value) and "gate_up_proj" in str(error.value)
