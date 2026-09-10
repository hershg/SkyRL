"""GLM-5.3-Flash sparse attention on top of megatron-core's ``DSAttention``.

The pinned Megatron candidate implements GLM's K-pool indexer. It performs
pool-level top-k selection with token expansion and retains an incomplete
causal tail, so this subclass intentionally adds no sequence-length guard.
"""

from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention


class Glm5NextDSAttention(DSAttention):
    """GLM-5.3-Flash NoPE MLA plus the pinned K-pool DSA implementation."""
