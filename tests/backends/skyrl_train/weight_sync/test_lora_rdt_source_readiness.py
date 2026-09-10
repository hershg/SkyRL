from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_rdt import (
    publication as publication_module,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.contracts import (
    LoRAAdapterLayout,
    LoRATensorSlice,
    LoRAUpdateRequest,
)
from skyrl.backends.skyrl_train.weight_sync.lora_rdt.producer import LoRardtProducer


@pytest.fixture
def published_rank(monkeypatch):
    layout = LoRAAdapterLayout(
        "adapter",
        tensors=(LoRATensorSlice("a", (2,), 0, 0, 0, 0, 8),),
        source_dtype="float32",
    )
    request = LoRAUpdateRequest.from_layout(layout, 4)
    producer = LoRardtProducer(0, layout)
    actor = SimpleNamespace(
        publish=SimpleNamespace(remote=producer.publish),
        discard=SimpleNamespace(remote=producer.discard),
    )
    publication = SimpleNamespace(
        request=request,
        local_tensors={"a": torch.tensor([1.0, 2.0])},
        rendezvous=SimpleNamespace(consumer_count=1),
    )
    monkeypatch.setattr(publication_module.ray, "get", lambda result, timeout: result)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    return producer, actor, publication


def test_success_is_reported_only_after_all_rank_readiness(published_rank, monkeypatch):
    producer, actor, publication = published_rank
    gathered = []

    def gather(results, receipt):
        assert producer.retained_generations() == [4]
        gathered.append(receipt)
        results[:] = [receipt, receipt]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    publication_module.publish_lora_sources(actor, publication, 2)

    assert gathered == [(4, publication.request.layout_digest, None)]
    assert torch.equal(producer.pull(4, ["a"])["a"], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize(
    "remote_generation,remote_error", [(4, "source failed"), (5, None)]
)
def test_other_rank_failure_or_generation_mismatch_releases_local_sources(
    published_rank, monkeypatch, remote_generation, remote_error
):
    producer, actor, publication = published_rank

    def gather(results, receipt):
        if isinstance(receipt, tuple):
            results[:] = [
                receipt,
                (remote_generation, publication.request.layout_digest, remote_error),
            ]
        else:
            results[:] = [receipt, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="sources not ready"):
        publication_module.publish_lora_sources(actor, publication, 2)

    assert producer.retained_generations() == []
    with pytest.raises(ValueError, match="not retained"):
        producer.pull(4, ["a"])


def test_failed_source_publish_still_joins_error_collective(
    published_rank, monkeypatch
):
    producer, actor, publication = published_rank
    publication.local_tensors["a"] = torch.ones(3)
    errors = []

    def gather(results, receipt):
        if isinstance(receipt, tuple):
            errors.append(receipt[2])
            results[:] = [receipt, (4, publication.request.layout_digest, None)]
        else:
            results[:] = [receipt, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="sources not ready"):
        publication_module.publish_lora_sources(actor, publication, 2)

    assert "shape" in errors[0]
    assert producer.retained_generations() == []
