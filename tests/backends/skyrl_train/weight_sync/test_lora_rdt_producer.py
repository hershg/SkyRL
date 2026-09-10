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
