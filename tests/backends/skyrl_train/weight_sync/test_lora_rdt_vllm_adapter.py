import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import build_vllm_lora_model

pytest.importorskip(
    "vllm", reason="lora_rdt receiver uses vLLM's pinned LoRAModel constructor"
)
pytestmark = pytest.mark.vllm


def _source_tensors():
    return {
        "base_model.model.layers.0.mlp.down_proj.lora_A.weight": torch.arange(
            8, dtype=torch.float32
        ).reshape(2, 4),
        "base_model.model.layers.0.mlp.down_proj.lora_B.weight": torch.arange(
            8, 16, dtype=torch.float32
        ).reshape(4, 2),
    }


def test_build_vllm_lora_model_materializes_independent_bf16_tensors(monkeypatch):
    from vllm.lora import lora_model

    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)
    source = _source_tensors()

    model = build_vllm_lora_model(
        adapter_id=7,
        adapter_config={"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]},
        source_tensors=source,
        device="cpu",
        dtype=torch.bfloat16,
    )

    layer = next(iter(model.loras.values()))
    assert model.id == 7
    assert layer.lora_a.dtype is torch.bfloat16
    assert layer.lora_b.dtype is torch.bfloat16
    assert (
        layer.lora_a.data_ptr()
        != source["base_model.model.layers.0.mlp.down_proj.lora_A.weight"].data_ptr()
    )
    assert (
        layer.lora_b.data_ptr()
        != source["base_model.model.layers.0.mlp.down_proj.lora_B.weight"].data_ptr()
    )
    layer.lora_a[0, 0] = -9
    assert (
        source["base_model.model.layers.0.mlp.down_proj.lora_A.weight"][0, 0].item()
        == 0
    )


def test_build_vllm_lora_model_rejects_non_fp32_sources():
    source = _source_tensors()
    source["base_model.model.layers.0.mlp.down_proj.lora_A.weight"] = source[
        "base_model.model.layers.0.mlp.down_proj.lora_A.weight"
    ].to(torch.bfloat16)

    with pytest.raises(ValueError, match="float32 source tensor"):
        build_vllm_lora_model(
            adapter_id=7,
            adapter_config={"r": 2, "lora_alpha": 2, "target_modules": ["down_proj"]},
            source_tensors=source,
            device="cpu",
            dtype=torch.bfloat16,
        )
