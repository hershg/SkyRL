import torch

from examples.model_checks.lora_logprobs import perturb_adapters, perturb_b_only


def test_b_only_matches_original_direction_without_touching_a():
    a = torch.nn.Parameter(torch.full((4, 8), 0.125, dtype=torch.bfloat16))
    b = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16))
    parameters = [("chunk0.adapter.linear_in.weight", a), ("chunk0.adapter.linear_out.weight", b)]
    expected_b = torch.nn.Parameter(torch.zeros_like(b))
    perturb_adapters([(parameters[1][0], expected_b)])
    expected = (expected_b.float() * 10).to(torch.bfloat16)
    report = perturb_b_only(parameters)
    assert torch.equal(a, torch.full_like(a, 0.125))
    assert torch.equal(b, expected)
    assert report["a_unchanged_tensors"] == 1 and report["b_multiplier"] == 10
