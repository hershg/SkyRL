import torch

from examples.model_checks.heldout_publication import (
    build_heldout_sequences,
    negate_adapter_b,
)


def test_heldout_fixture_is_fixed_and_not_truncated_or_padded():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [ord(char) for char in text]

    first, second = build_heldout_sequences(Tokenizer())
    assert len(first) == 97 and len(second) == 193
    assert first[:10] == [ord(char) for char in "A botanist"]
    assert second[:11] == [ord(char) for char in "The library"]


def test_wrong_adapter_reverses_only_b_without_changing_source():
    original = {
        "q.lora_A.weight": torch.ones(3),
        "q.lora_B.weight": torch.arange(3).float(),
    }
    wrong = negate_adapter_b(original)
    assert torch.equal(wrong["q.lora_A.weight"], original["q.lora_A.weight"])
    assert torch.equal(wrong["q.lora_B.weight"], -original["q.lora_B.weight"])
    assert torch.equal(original["q.lora_B.weight"], torch.arange(3).float())
