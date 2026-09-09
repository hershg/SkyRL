"""Deterministic adapter-only perturbation for native GPU checks."""

import hashlib

import torch

from skyrl.tinker.logprob_checks import (
    check_initial_adapter as check_initial_adapter,
    check_updated_adapter as check_updated_adapter,
    check_withheld_publication as check_withheld_publication,
    compare_logprobs as compare_logprobs,
)


@torch.no_grad()
def perturb_adapters(named_parameters, seed=0, scale=1e-3):
    """Use name-seeded noise so replicated adapter tensors receive identical updates."""
    changed = 0
    tensors = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        assert "adapter" in name or "lora" in name, f"unexpected trainable base parameter: {name}"
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
        generator = torch.Generator(device=parameter.device).manual_seed((seed + name_seed) % (2**63))
        parameter.add_(
            torch.randn(parameter.shape, generator=generator, device=parameter.device, dtype=parameter.dtype),
            alpha=scale,
        )
        changed += parameter.numel()
        tensors += 1
    assert changed > 0
    return {"trainable_tensors": tensors, "trainable_elements": changed, "seed": seed, "noise_std": scale}


@torch.no_grad()
def perturb_b_only(named_parameters):
    parameters = [(name, p) for name, p in named_parameters if p.requires_grad]
    a = [(name, p) for name, p in parameters if name.endswith(".linear_in.weight")]
    b = [(name, p) for name, p in parameters if name.endswith(".linear_out.weight")]
    assert a and b and len(a) + len(b) == len(parameters)
    preserved = [p.clone() for _, p in a]
    assert all(torch.count_nonzero(p) == 0 for _, p in b)
    report = perturb_adapters(b)
    for _, parameter in b:
        parameter.mul_(10)
    assert all(torch.equal(before, after) for before, (_, after) in zip(preserved, a))
    report.update(a_unchanged_tensors=len(a), b_multiplier=10)
    return report
