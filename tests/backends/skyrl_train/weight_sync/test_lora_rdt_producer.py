import gc
import logging
import weakref

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRAAdapterLayout,
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRardtProducer,
    LoRATensorSlice,
    LoRAUpdateRequest,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.contracts import LoRASourceSlice


@pytest.fixture
def layout():
    return LoRAAdapterLayout(
        adapter_name="adapter",
        source_dtype="float32",
        tensors=(
            LoRATensorSlice("a", (2,), 0, 0, 0, 0, 8),
            LoRATensorSlice("b", (2,), 1, 0, 0, 8, 8),
        ),
    )


def test_producer_serves_only_its_rank_owned_tensors_until_all_consumers_acknowledge(
    layout,
):
    producer = LoRardtProducer(source_rank=0, layout=layout)
    request = LoRAUpdateRequest.from_layout(layout, generation=4)
    source = torch.tensor([1.0, 2.0])

    producer.publish(request, {"a": source}, consumer_count=2)

    assert producer.pull(4, ["a"]) == {"a": source}
    assert producer.acknowledge(4, consumer_id=3) is False
    assert producer.retained_generations() == [4]
    assert producer.acknowledge(4, consumer_id=4) is True
    assert producer.retained_generations() == []


def test_producer_rejects_wrong_owner_stale_generation_and_duplicate_acknowledgement(
    layout,
):
    producer = LoRardtProducer(source_rank=0, layout=layout)
    request = LoRAUpdateRequest.from_layout(layout, generation=4)
    producer.publish(request, {"a": torch.tensor([1.0, 2.0])}, consumer_count=2)

    with pytest.raises(ValueError, match="does not own"):
        producer.pull(4, ["b"])
    assert producer.acknowledge(4, consumer_id=3) is False
    assert producer.acknowledge(4, consumer_id=3) is False
    producer.discard(4)
    with pytest.raises(ValueError, match="stale"):
        producer.publish(
            LoRAUpdateRequest.from_layout(layout, generation=3),
            {"a": torch.tensor([1.0, 2.0])},
            2,
        )


def test_bridge_source_producer_validates_its_rank_local_fp32_snapshot():
    key = "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight"
    layout = LoRABridgeSourceLayout(
        "adapter",
        (
            LoRABridgeSource(
                key=key,
                source_rank=0,
                hf_param_names=("down_proj.lora_B.weight",),
                component="linear_out",
                transform="identity",
                shape=(1, 2),
                tensor_parallel_axis=0,
                tensor_parallel_rank=0,
                tensor_parallel_size=2,
                expert_parallel_axis=None,
                expert_parallel_rank=0,
                expert_parallel_size=1,
                transform_config=(),
            ),
            LoRABridgeSource(
                key=key,
                source_rank=1,
                hf_param_names=("down_proj.lora_B.weight",),
                component="linear_out",
                transform="identity",
                shape=(1, 2),
                tensor_parallel_axis=0,
                tensor_parallel_rank=1,
                tensor_parallel_size=2,
                expert_parallel_axis=None,
                expert_parallel_rank=0,
                expert_parallel_size=1,
                transform_config=(),
            ),
        ),
    )
    request = LoRAUpdateRequest("adapter", 4, layout.layout_digest, "float32")
    producer = LoRardtProducer(source_rank=0, layout=layout)

    producer.publish(request, {key: torch.tensor([[1.0, 2.0]])}, consumer_count=1)

    assert producer.pull(4, [key])[key].dtype is torch.float32
    with pytest.raises(ValueError, match="shape"):
        LoRardtProducer(source_rank=1, layout=layout).publish(
            request,
            {key: torch.tensor([1.0, 2.0])},
            consumer_count=1,
        )


def test_discard_before_delayed_publish_prevents_generation_resurrection(layout):
    producer = LoRardtProducer(0, layout)
    request = LoRAUpdateRequest.from_layout(layout, 4)

    producer.discard(4)
    producer.discard(4)
    with pytest.raises(ValueError, match="stale"):
        producer.publish(request, {"a": torch.tensor([1.0, 2.0])}, 1)

    assert producer.retained_generations() == []
    producer.publish(LoRAUpdateRequest.from_layout(layout, 5), {"a": torch.ones(2)}, 1)
    assert producer.retained_generations() == [5]


def test_discard_serializes_with_inflight_publish(layout, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    producer = LoRardtProducer(0, layout)
    request = LoRAUpdateRequest.from_layout(layout, 4)
    entered = Event()
    release = Event()
    discard_started = Event()
    validate = producer._validate_request

    def delayed_validate(value):
        validate(value)
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release publication")

    def discard():
        discard_started.set()
        producer.discard(4)

    monkeypatch.setattr(producer, "_validate_request", delayed_validate)
    with ThreadPoolExecutor(max_workers=2) as executor:
        publication = executor.submit(producer.publish, request, {"a": torch.ones(2)}, 1)
        try:
            assert entered.wait(5)
            cleanup = executor.submit(discard)
            assert discard_started.wait(5)
        finally:
            release.set()
        publication.result(timeout=5)
        cleanup.result(timeout=5)

    assert producer.retained_generations() == []


def test_slice_pull_packs_only_consumed_fp32_bytes_and_releases_after_last_ack(caplog):
    caplog.set_level(logging.INFO, logger="skyrl.backends.skyrl_train.weight_sync.lora_rdt.producer")
    layout = LoRAAdapterLayout("adapter", "float32", (LoRATensorSlice("a", (4, 16), 0, 0, 0, 0, 256),))
    producer = LoRardtProducer(0, layout)
    source = torch.arange(64, dtype=torch.float32).reshape(4, 16) + 0.03125
    producer.publish(LoRAUpdateRequest.from_layout(layout, 1), {"a": source}, 2)
    selections = [
        LoRASourceSlice("a", (1, 0), (3, 16)),
        LoRASourceSlice("a", (0, 5), (4, 7)),
    ]
    packed = producer.pull_slices(1, selections)
    for selection, tensor in zip(selections, packed):
        assert torch.equal(tensor, source[selection.indices])
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous() and tensor.storage_offset() == 0
        assert tensor.untyped_storage().nbytes() == tensor.numel() * 4
        assert tensor.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert sum(tensor.numel() * 4 for tensor in packed) == 160
    repeated = producer.pull_slices(1, selections)
    assert all(first is second for first, second in zip(packed, repeated))
    receipts = [record.message for record in caplog.records if "lora_rdt_producer_stage" in record.message]
    assert "new_slices=2 new_bytes=160" in receipts[0]
    assert "new_slices=0 new_bytes=0" in receipts[1]
    references = [weakref.ref(tensor) for tensor in packed]
    del packed, repeated, tensor
    gc.collect()
    assert not producer.acknowledge(1, 0)
    assert all(reference() is not None for reference in references)
    assert producer.acknowledge(1, 1)
    gc.collect()
    assert all(reference() is None for reference in references)
    with pytest.raises(ValueError, match="not retained"):
        producer.pull_slices(1, selections)


@pytest.mark.parametrize(
    "selection",
    [
        LoRASourceSlice("missing", (0,), (1,)),
        LoRASourceSlice("a", (0, 0), (1, 1)),
        LoRASourceSlice("a", (0,), (3,)),
    ],
)
def test_invalid_slice_pull_does_not_change_published_source(layout, selection):
    producer = LoRardtProducer(0, layout)
    source = torch.tensor([1.03125, 2.03125])
    producer.publish(LoRAUpdateRequest.from_layout(layout, 1), {"a": source}, 1)
    with pytest.raises(ValueError, match="does not own|exceeds source shape"):
        producer.pull_slices(1, [LoRASourceSlice("a", (0,), (1,)), selection])
    assert torch.equal(producer.pull(1, ["a"])["a"], source)
    assert producer.retained_generations() == [1]
    producer.discard(1)
    with pytest.raises(ValueError, match="not retained"):
        producer.pull_slices(1, [LoRASourceSlice("a", (0,), (1,))])
