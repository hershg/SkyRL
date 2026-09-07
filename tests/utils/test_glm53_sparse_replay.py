import pytest
import torch

from tests.utils.glm53_sparse_replay import compare_tensors, compute_reference


def make_capture():
    return {
        "query": torch.tensor([[[[1.0, 0.0]]]]),
        "kv_cache": torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]]]),
        "block_tables": torch.tensor([[[1, 0, -1]]]),
        "seq_lens": torch.tensor([2]),
        "scalars": {"bmm1_scale": 0.5, "bmm2_scale": 2.0, "kv_lora_rank": 1},
    }


def test_reference_applies_mask_scales_and_latent_value_slice():
    capture = make_capture()
    probabilities = torch.tensor([1.5, 0.5]).softmax(dim=0)
    expected = 2 * (probabilities[0] * 3 + probabilities[1])
    actual = compute_reference(capture)
    assert actual.shape == (1, 1, 1, 1)
    torch.testing.assert_close(actual.flatten()[0], expected)
    capture["block_tables"] = torch.tensor([[[0, 1, -1]]])
    torch.testing.assert_close(compute_reference(capture), actual)


def test_reference_rejects_inconsistent_valid_counts():
    capture = make_capture()
    capture["seq_lens"] = torch.tensor([3])
    with pytest.raises(AssertionError):
        compute_reference(capture)


def test_comparison_reports_changes_and_rejects_nonfinite_values():
    reference = torch.tensor([1.0, 2.0])
    result = compare_tensors(torch.tensor([1.0, 3.0]), reference)
    assert result["different_elements"] == 1
    assert result["mean"] == 0.5
    assert result["max"] == 1.0
    assert result["relative_rms"] == pytest.approx(1 / 5**0.5)
    with pytest.raises(AssertionError):
        compare_tensors(torch.tensor([float("nan"), 2.0]), reference)
