"""Independent BF16/FP32 scoring of the exact predeclared native Qwen8 export."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from examples.model_checks.run_hf_lora_reference import run_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ready = json.loads((args.native_output / "reference-ready.json").read_text())
    adapter = args.native_output / "updated-export"
    with (adapter / "adapter_model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == ready["adapter_sha256"]
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "diagnostic_only": True,
        "adapter_sha256": digest,
        "minimum_signal_error_ratio": 5,
        "fixtures": {},
    }
    for label, tokens in ready["tokens"].items():
        fixture = {"tokens": tokens}
        for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            fixture[name] = run_reference(args.model, adapter, tokens, dtype)
        bf16 = torch.tensor(fixture["bf16"]["updated"], dtype=torch.float64) - torch.tensor(
            fixture["bf16"]["base"], dtype=torch.float64
        )
        fp32 = torch.tensor(fixture["fp32"]["updated"], dtype=torch.float64) - torch.tensor(
            fixture["fp32"]["base"], dtype=torch.float64
        )
        signal = fp32.abs().mean().item()
        error = (bf16 - fp32).abs().mean().item()
        assert error > 0 and torch.isfinite(bf16).all() and torch.isfinite(fp32).all()
        fixture.update(
            signal_mean_abs=signal,
            arithmetic_error_mean_abs=error,
            signal_error_ratio=signal / error,
            signal_separation_passed=signal / error >= 5,
        )
        report["fixtures"][label] = fixture
        pending = args.output.with_suffix(".json.tmp")
        pending.write_text(json.dumps(report, indent=2, allow_nan=False))
        pending.replace(args.output)
    assert all(fixture["signal_separation_passed"] for fixture in report["fixtures"].values())


if __name__ == "__main__":
    main()
