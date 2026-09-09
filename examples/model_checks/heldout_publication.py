"""Fixed diagnostic controls, independent of the original two calibration prompts."""

from itertools import cycle, islice


def build_heldout_sequences(tokenizer):
    return [
        list(islice(cycle(tokenizer.encode(text, add_special_tokens=False)), length))
        for text, length in [
            ("A botanist records the color of each flower in a shaded garden. ", 97),
            (
                "The library closes at six; return three books before taking the evening train. ",
                193,
            ),
        ]
    ]


def negate_adapter_b(tensors):
    assert tensors and all(".lora_A." in name or ".lora_B." in name for name in tensors)
    return {name: -value if ".lora_B." in name else value.clone() for name, value in tensors.items()}
