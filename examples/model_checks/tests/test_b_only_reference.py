import pytest
import torch

from examples.model_checks.active_lora_audit import build_b_only_candidate


def test_b_only_candidate_preserves_a_and_scales_direction():
    zero = {"q.lora_A.weight": torch.arange(4, dtype=torch.float32), "q.lora_B.weight": torch.zeros(4)}
    direction = {"q.lora_A.weight": torch.ones(4), "q.lora_B.weight": torch.arange(4, dtype=torch.float32)}
    result, norms = build_b_only_candidate(zero, direction)
    assert torch.equal(result["q.lora_A.weight"], zero["q.lora_A.weight"])
    assert torch.equal(result["q.lora_B.weight"], direction["q.lora_B.weight"] * 10)
    assert norms["q.lora_B.weight"]["candidate"] == pytest.approx(norms["q.lora_B.weight"]["original"] * 10)
    assert torch.count_nonzero(zero["q.lora_B.weight"]) == 0


def test_nonzero_baseline_b_rejected():
    with pytest.raises(AssertionError):
        build_b_only_candidate({"q.lora_B.weight": torch.ones(4)}, {"q.lora_B.weight": torch.ones(4)})
