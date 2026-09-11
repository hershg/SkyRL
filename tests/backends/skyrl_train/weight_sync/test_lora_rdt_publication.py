from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt.publication import (
    LoRardtPublicationPlanner,
    make_lora_rdt_producer_name,
)


def _record(weight):
    return SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight",
        weight=weight,
        hf_param_names=("down_proj.lora_B.weight",),
        component="linear_out",
        transform="identity",
        tensor_parallel_axis=0,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=(),
    )


def test_publication_planner_keeps_layout_and_advances_generation():
    planner = LoRardtPublicationPlanner("adapter", 0, 1, "test")
    records = [_record(torch.ones((2, 1), dtype=torch.float32))]
    first = planner.plan(records, [records_to_sources(records)], ["producer-0"])
    second = planner.plan(records)

    assert first.request.generation == 0
    assert second.request.generation == 1
    assert first.layout.layout_digest == second.layout.layout_digest
    assert first.rendezvous.actor_name_by_rank() == {0: "producer-0"}
    assert make_lora_rdt_producer_name(first.layout, 0).endswith("_rk0")


def test_publication_planner_rejects_layout_change():
    planner = LoRardtPublicationPlanner("adapter", 0, 1, None)
    first = [_record(torch.ones((2, 1), dtype=torch.float32))]
    planner.plan(first, [records_to_sources(first)], ["producer-0"])
    changed = [_record(torch.ones((3, 1), dtype=torch.float32))]

    with pytest.raises(ValueError, match="changed its fixed local source layout"):
        planner.plan(changed)


def test_publication_planner_requires_static_metadata_once():
    planner = LoRardtPublicationPlanner("adapter", 0, 1, None)
    records = [_record(torch.ones((2, 1), dtype=torch.float32))]

    with pytest.raises(ValueError, match="first publication requires"):
        planner.plan(records)


def test_publication_planner_rejects_reinitialization():
    planner = LoRardtPublicationPlanner("adapter", 0, 1, None)
    records = [_record(torch.ones((2, 1), dtype=torch.float32))]
    sources = [records_to_sources(records)]
    planner.plan(records, sources, ["producer-0"])

    with pytest.raises(ValueError, match="already initialized"):
        planner.plan(records, sources, ["producer-0"])


def records_to_sources(records):
    from skyrl.backends.skyrl_train.weight_sync.lora_rdt.bridge_sources import (
        extract_lora_bridge_sources,
    )

    return extract_lora_bridge_sources(records)[1]
