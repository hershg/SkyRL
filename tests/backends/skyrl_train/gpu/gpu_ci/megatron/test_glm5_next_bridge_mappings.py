"""GLM-5.3-Flash bridge mapping contracts against Megatron Bridge."""

import pytest
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.bridge import Glm5NextBridge

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
