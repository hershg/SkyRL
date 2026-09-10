import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRAAdapterGenerationState,
    LoRAAdapterLayout,
    LoRATensorSlice,
    LoRATransferInitInfo,
    LoRAUpdateRequest,
    build_lora_adapter_layout,
    materialize_bf16_adapter_tensor,
)


def _layout() -> LoRAAdapterLayout:
    return LoRAAdapterLayout(
        adapter_name="glm53-rank32",
        source_dtype="float32",
        tensors=(
            LoRATensorSlice(
                key="base_model.model.layers.0.mlp.down_proj.lora_A.weight",
                shape=(2, 4),
                source_rank=0,
                source_offset=0,
                destination_rank=0,
                destination_offset=0,
                byte_length=32,
            ),
            LoRATensorSlice(
                key="base_model.model.layers.0.mlp.down_proj.lora_B.weight",
                shape=(4, 2),
                source_rank=1,
                source_offset=0,
                destination_rank=1,
                destination_offset=0,
                byte_length=32,
            ),
        ),
    )


def _state() -> LoRAAdapterGenerationState:
    return LoRAAdapterGenerationState(
        LoRATransferInitInfo(layout=_layout(), inference_ranks=(0, 1))
    )


def test_layout_digest_is_content_independent_and_ownership_sensitive():
    layout = _layout()
    same_layout = _layout()
    changed_ownership = LoRAAdapterLayout(
        adapter_name=layout.adapter_name,
        source_dtype=layout.source_dtype,
        tensors=(
            layout.tensors[0],
            LoRATensorSlice(
                key=layout.tensors[1].key,
                shape=layout.tensors[1].shape,
                source_rank=0,
                source_offset=0,
                destination_rank=1,
                destination_offset=0,
                byte_length=32,
            ),
        ),
    )

    assert layout.layout_digest == same_layout.layout_digest
    assert layout.layout_digest != changed_ownership.layout_digest


def test_layout_builder_uses_canonical_keys_and_rank_local_offsets_without_reading_values():
    first = torch.zeros((2, 4), dtype=torch.float32)
    second = torch.ones((4, 2), dtype=torch.float32)
    layout = build_lora_adapter_layout(
        "glm53-rank32",
        {
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight": second,
            "base_model.model.layers.0.mlp.down_proj.lora_A.weight": first,
        },
        {
            "base_model.model.layers.0.mlp.down_proj.lora_A.weight": (4, 1),
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight": (4, 1),
        },
    )

    assert [tensor.key for tensor in layout.tensors] == sorted(
        [
            "base_model.model.layers.0.mlp.down_proj.lora_A.weight",
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight",
        ]
    )
    assert [tensor.source_offset for tensor in layout.tensors] == [0, 32]
    assert [tensor.destination_offset for tensor in layout.tensors] == [0, 32]
    changed_values = build_lora_adapter_layout(
        layout.adapter_name,
        {
            "base_model.model.layers.0.mlp.down_proj.lora_A.weight": torch.full(
                (2, 4), 9.0
            ),
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight": torch.full(
                (4, 2), -4.0
            ),
        },
        {
            "base_model.model.layers.0.mlp.down_proj.lora_A.weight": (4, 1),
            "base_model.model.layers.0.mlp.down_proj.lora_B.weight": (4, 1),
        },
    )

    assert changed_values.layout_digest == layout.layout_digest


def test_layout_rejects_unsorted_keys_and_wrong_float32_byte_length():
    layout = _layout()

    with pytest.raises(ValueError, match="sorted and unique"):
        LoRAAdapterLayout(
            adapter_name=layout.adapter_name,
            source_dtype="float32",
            tensors=tuple(reversed(layout.tensors)),
        )
    with pytest.raises(ValueError, match="expected 32"):
        LoRAAdapterLayout(
            adapter_name=layout.adapter_name,
            source_dtype="float32",
            tensors=(
                LoRATensorSlice(
                    key=layout.tensors[0].key,
                    shape=layout.tensors[0].shape,
                    source_rank=0,
                    source_offset=0,
                    destination_rank=0,
                    destination_offset=0,
                    byte_length=16,
                ),
            ),
        )


def test_generation_activates_only_after_all_ranks_acknowledge():
    state = _state()
    request = LoRAUpdateRequest.from_layout(_layout(), generation=3)
    rank_zero_staging = object()
    rank_one_staging = object()

    state.stage(request, 0, rank_zero_staging)
    with pytest.raises(ValueError, match=r"ranks \[1\]"):
        state.activate(request)

    state.stage(request, 1, rank_one_staging)

    assert state.activate(request) == {0: rank_zero_staging, 1: rank_one_staging}
    assert state.active_generation == 3


def test_failed_generation_discards_staging_and_leaves_previous_generation_active():
    state = _state()
    first = LoRAUpdateRequest.from_layout(_layout(), generation=0)
    state.stage(first, 0, "generation-zero-rank-zero")
    state.stage(first, 1, "generation-zero-rank-one")
    state.activate(first)
    second = LoRAUpdateRequest.from_layout(_layout(), generation=1)
    state.stage(second, 0, "generation-one-rank-zero")

    state.discard(second)

    assert state.active_generation == 0
    assert state.active_buffers == {
        0: "generation-zero-rank-zero",
        1: "generation-zero-rank-one",
    }
    with pytest.raises(ValueError, match=r"ranks \[0, 1\]"):
        state.activate(second)


def test_generation_rejects_stale_and_mismatched_layout_requests():
    layout = _layout()
    state = _state()
    current = LoRAUpdateRequest.from_layout(layout, generation=2)
    state.stage(current, 0, object())
    state.stage(current, 1, object())
    state.activate(current)

    with pytest.raises(ValueError, match="stale"):
        state.stage(LoRAUpdateRequest.from_layout(layout, generation=1), 0, object())
    with pytest.raises(ValueError, match="layout digest"):
        state.stage(
            LoRAUpdateRequest(
                adapter_name=layout.adapter_name,
                generation=3,
                layout_digest="0" * 64,
                source_dtype="float32",
            ),
            0,
            object(),
        )


def test_materialized_bf16_adapter_storage_is_independent_of_fp32_source():
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    destination = materialize_bf16_adapter_tensor(source)
    destination[0, 0] = -7

    assert destination.dtype is torch.bfloat16
    assert destination.data_ptr() != source.data_ptr()
    assert source[0, 0].item() == 0
