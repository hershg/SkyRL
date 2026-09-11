from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)


@pytest.mark.asyncio
async def test_lora_rdt_reuses_static_publication_state(monkeypatch):
    from megatron.bridge.models.conversion import peft_bridge

    from skyrl.backends.skyrl_train.inference_servers import remote_inference_client
    from skyrl.backends.skyrl_train.weight_sync.lora_rdt import publication
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker

    record = SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight",
        weight=torch.ones((2, 1), dtype=torch.float32),
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
    worker = object.__new__(MegatronPolicyWorkerBase)
    worker.actor_module = object()
    worker.bridge = SimpleNamespace(
        export_local_adapter_weights=lambda actor_module: [record]
    )
    worker.lora_cls = object()
    worker._logical_model_path = "model"

    class RemoteClient:
        server_urls = ("server",)

        def __init__(self):
            self.generations = []

        async def load_lora_rdt_adapter(
            self, lora_name, rendezvous, request, adapter_config
        ):
            assert lora_name == "adapter"
            assert rendezvous["layout"]["adapter_name"] == "adapter"
            assert adapter_config == {"target_modules": ["down_proj"]}
            self.generations.append(request["generation"])

    monkeypatch.setattr(remote_inference_client, "RemoteInferenceClient", RemoteClient)
    monkeypatch.setattr(
        peft_bridge,
        "infer_target_modules_from_adapter_weights",
        lambda weights: ["down_proj"],
    )
    monkeypatch.setattr(
        peft_bridge,
        "build_adapter_config_dict",
        lambda lora_cls, target_modules, base_model_name_or_path: {
            "target_modules": target_modules
        },
    )

    acknowledgements = []
    actor = SimpleNamespace(
        acknowledge=SimpleNamespace(
            remote=lambda generation, consumer_id: acknowledgements.append(
                (generation, consumer_id)
            )
        ),
        discard=SimpleNamespace(remote=lambda generation: None),
    )
    actor_lookups = []
    monkeypatch.setattr(
        megatron_worker.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(namespace="test"),
    )
    monkeypatch.setattr(
        megatron_worker.ray,
        "get_actor",
        lambda name, namespace: actor_lookups.append((name, namespace)) or actor,
    )
    monkeypatch.setattr(megatron_worker.ray, "get", lambda values: values)

    gathers = []

    def all_gather(results, value):
        gathers.append(value)
        results[:] = [value]

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda values, src: None
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    publications = []
    monkeypatch.setattr(
        publication,
        "publish_lora_sources",
        lambda producer, update, timeout: publications.append(
            (producer, update.request.generation)
        ),
    )
    client = RemoteClient()

    await worker._publish_lora_rdt_adapter(client, "adapter")
    await worker._publish_lora_rdt_adapter(client, "adapter")

    assert len(gathers) == 2
    assert len(actor_lookups) == 2
    assert publications == [(actor, 0), (actor, 1)]
    assert acknowledgements == [(0, 0), (1, 0)]
    assert client.generations == [0, 1]
