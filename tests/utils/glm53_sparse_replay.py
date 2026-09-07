"""Reference and repeat comparisons for captured sparse MLA rows."""

import torch


def compute_reference(capture):
    query = capture["query"].float().squeeze(1)
    indices = capture["block_tables"].long().squeeze(1)
    valid = indices >= 0
    assert valid.any(dim=-1).all()
    assert torch.equal(valid.sum(dim=-1), capture["seq_lens"].reshape(-1).long())
    cache = capture["kv_cache"].float().reshape(-1, query.shape[-1])
    keys = cache[indices.clamp_min(0)]
    logits = torch.einsum("bhd,bkd->bhk", query, keys)
    logits *= capture["scalars"]["bmm1_scale"]
    logits.masked_fill_(~valid[:, None], float("-inf"))
    probabilities = logits.softmax(dim=-1)
    values = keys[..., : capture["scalars"]["kv_lora_rank"]]
    output = torch.einsum("bhk,bkd->bhd", probabilities, values)
    return (output * capture["scalars"]["bmm2_scale"]).unsqueeze(1)


def compare_tensors(left, right):
    assert left.shape == right.shape
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    difference = (left.float() - right.float()).abs().flatten()
    reference_rms = right.float().square().mean().sqrt().item()
    assert reference_rms > 0
    return {
        "elements": difference.numel(),
        "mean": difference.mean().item(),
        "max": difference.max().item(),
        "p99": torch.quantile(difference, 0.99).item(),
        "relative_rms": difference.square().mean().sqrt().item() / reference_rms,
        "different_elements": torch.count_nonzero(difference).item(),
    }
