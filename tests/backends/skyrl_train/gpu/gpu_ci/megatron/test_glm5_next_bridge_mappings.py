"""GLM-5.3-Flash bridge mapping contracts against Megatron Bridge."""

import pytest
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.bridge import Glm5NextBridge
from skyrl.backends.skyrl_train.workers.megatron.glm5_next.provider import (
    Glm5NextModelProvider,
)

pytestmark = pytest.mark.megatron


def test_dsa_indexer_uses_explicit_replicated_mappings():
    registry = Glm5NextBridge.mapping_registry(None)
    indexer_mappings = [
        mapping
        for mapping in registry.mappings
        if ".core_attention.indexer." in str(mapping.megatron_param)
    ]

    assert len(indexer_mappings) == 7
    assert all(isinstance(mapping, ReplicatedMapping) for mapping in indexer_mappings)


def test_model_provider_uses_sparse_dsa_kernel():
    provider = Glm5NextModelProvider(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=8,
    )

    assert provider.dsa_kernel_backend == "tilelang"
