import pytest

import skyrl.backends.skyrl_train.weight_sync.lora_rdt.receiver as receiver
from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRardtProducerRendezvous,
    resolve_lora_rdt_producers,
)


def _layout():
    key = "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight"
    return LoRABridgeSourceLayout(
        "adapter",
        (
            LoRABridgeSource(
                key=key,
                source_rank=3,
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
                source_rank=7,
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


def test_rendezvous_round_trips_and_resolves_named_producers(monkeypatch):
    rendezvous = LoRardtProducerRendezvous(
        layout=_layout(),
        producer_actor_names=((3, "lora-rdt-3"), (7, "lora-rdt-7")),
        consumer_count=2,
        namespace="skyrl",
    )
    calls = []
    monkeypatch.setattr(
        receiver.ray,
        "get_actor",
        lambda name, namespace: calls.append((name, namespace)) or name,
    )

    restored = LoRardtProducerRendezvous.from_json_dict(rendezvous.to_json_dict())

    assert restored == rendezvous
    assert resolve_lora_rdt_producers(
        restored.actor_name_by_rank(), restored.namespace
    ) == {
        3: "lora-rdt-3",
        7: "lora-rdt-7",
    }
    assert calls == [("lora-rdt-3", "skyrl"), ("lora-rdt-7", "skyrl")]


def test_rendezvous_rejects_missing_source_rank_or_duplicate_actor_name():
    with pytest.raises(ValueError, match="producer ranks"):
        LoRardtProducerRendezvous(
            layout=_layout(),
            producer_actor_names=((3, "lora-rdt-3"),),
            consumer_count=1,
        )
    with pytest.raises(ValueError, match="unique"):
        LoRardtProducerRendezvous(
            layout=_layout(),
            producer_actor_names=((3, "same"), (7, "same")),
            consumer_count=1,
        )
