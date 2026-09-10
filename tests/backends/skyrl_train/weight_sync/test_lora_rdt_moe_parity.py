from types import SimpleNamespace

import torch

import skyrl.backends.skyrl_train.weight_sync.lora_rdt.receiver as receiver
from skyrl.backends.skyrl_train.weight_sync.lora_layout import (
    convert_moe_expert_lora_key,
    convert_moe_experts_lora_to_vllm,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRAUpdateRequest,
)


def _create_expert_sources():
    shapes = {
        "gate_up_proj.lora_A": (4, 2, 3),
        "gate_up_proj.lora_B": (4, 8, 2),
        "down_proj.lora_A": (4, 2, 4),
        "down_proj.lora_B": (4, 3, 2),
    }
    full_tensors = {}
    sources = []
    shards = {0: {}, 1: {}}
    for index, (suffix, shape) in enumerate(shapes.items()):
        name = f"model.layers.0.mlp.experts.{suffix}.weight"
        tensor = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape) + 1000 * (index + 1)
        full_tensors[f"base_model.model.{name}"] = tensor
        for ep_rank, shard in enumerate(tensor.chunk(2, dim=0)):
            shards[ep_rank][name] = shard.clone()
            sources.append(
                LoRABridgeSource(
                    key=name,
                    source_rank=ep_rank,
                    hf_param_names=(name,),
                    component="linear_in" if suffix.endswith("lora_A") else "linear_out",
                    transform="identity",
                    shape=tuple(shard.shape),
                    tensor_parallel_axis=None,
                    tensor_parallel_rank=0,
                    tensor_parallel_size=1,
                    expert_parallel_axis=0,
                    expert_parallel_rank=ep_rank,
                    expert_parallel_size=2,
                    transform_config=(),
                )
            )
    sources.sort(key=lambda source: (source.key, source.expert_parallel_rank))
    return full_tensors, shards, LoRABridgeSourceLayout("adapter", tuple(sources))


def test_fused_expert_flat_layout_has_exact_expert_axis_order():
    full_tensors, _, _ = _create_expert_sources()

    converted = convert_moe_experts_lora_to_vllm(full_tensors)

    for name, source in full_tensors.items():
        key = convert_moe_expert_lora_key(name, source.ndim)
        actual = converted[key]
        if name.endswith(".lora_A.weight"):
            for expert in range(4):
                assert torch.equal(actual[expert * 2 : (expert + 1) * 2], source[expert])
        else:
            for expert in range(4):
                for rank in range(2):
                    assert torch.equal(actual[:, rank * 4 + expert], source[expert, :, rank])
    assert set(converted) == {
        f"base_model.model.model.layers.0.mlp.{module}.lora_{component}.weight"
        for module in ("experts", "experts.base_layer")
        for component in ("A", "B")
    }


def test_remote_expert_assembly_matches_disk_fp32_bytes_and_vllm_bf16_values(monkeypatch, tmp_path):
    from vllm.lora import lora_model
    from vllm.lora.peft_helper import PEFTHelper

    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)
    full_tensors, shards, layout = _create_expert_sources()
    disk_tensors = convert_moe_experts_lora_to_vllm(full_tensors)
    torch.save(disk_tensors, tmp_path / "adapter_model.bin")
    config = {"r": 2, "lora_alpha": 2, "target_modules": ["experts"]}
    disk_model = lora_model.LoRAModel.from_local_checkpoint(
        str(tmp_path),
        expected_lora_modules={"experts"},
        peft_helper=PEFTHelper.from_dict(config),
        lora_model_id=7,
        device="cpu",
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(receiver.ray, "get", lambda refs: refs)
    producers = {
        rank: SimpleNamespace(
            pull=SimpleNamespace(remote=lambda generation, names, values=values: {name: values[name] for name in names})
        )
        for rank, values in shards.items()
    }
    received = {}
    build_model = receiver.build_vllm_lora_model

    def build_and_check_model(**kwargs):
        received.update(kwargs["source_tensors"])
        return build_model(**kwargs)

    monkeypatch.setattr(receiver, "build_vllm_lora_model", build_and_check_model)
    staged = []
    monkeypatch.setattr(receiver, "stage_vllm_lora_model", lambda runner, model: staged.append(model))
    remote_model = receiver.pull_reconstruct_and_stage_lora_adapter(
        producers,
        layout,
        LoRAUpdateRequest.from_layout(layout, 1),
        8,
        config,
        object(),
        "cpu",
    )

    assert staged == [remote_model]
    assert received.keys() == disk_tensors.keys()
    for name, expected in disk_tensors.items():
        actual = received[name]
        assert actual.dtype is torch.float32
        assert actual.shape == expected.shape
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert remote_model.loras.keys() == disk_model.loras.keys()
    assert set(remote_model.loras) == {
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.base_layer",
    }
    for name, disk_layer in disk_model.loras.items():
        remote_layer = remote_model.loras[name]
        for attribute in ("lora_a", "lora_b"):
            actual = getattr(remote_layer, attribute)
            expected = getattr(disk_layer, attribute)
            assert actual.dtype is torch.bfloat16
            assert torch.equal(actual, expected)
            assert actual.data_ptr() != expected.data_ptr()
            before = expected.clone()
            actual.zero_()
            assert torch.equal(expected, before)
    for name, tensor in full_tensors.items():
        assert torch.count_nonzero(tensor) == tensor.numel()
    for rank_shards in shards.values():
        for tensor in rank_shards.values():
            assert torch.count_nonzero(tensor) == tensor.numel()
