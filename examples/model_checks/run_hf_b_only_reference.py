"""Diagnostic-only, predeclared B-only 10x stimulus; no acceptance-budget changes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from examples.model_checks.active_lora_audit import build_b_only_candidate
from examples.model_checks.heldout_publication import build_heldout_sequences
from examples.model_checks.run_hf_lora_reference import run_reference
from skyrl.utils.tok import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument(
        "--trainer-representable",
        action="store_true",
        help="Round B to native BF16 storage before export",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    zero = load_file(args.zero / "adapter_model.safetensors")
    direction = load_file(args.direction / "adapter_model.safetensors")
    candidate, norms = build_b_only_candidate(zero, direction)
    if args.trainer_representable:
        candidate = {name: value.to(torch.bfloat16).float() for name, value in candidate.items()}
        assert all(torch.equal(value, zero[name]) for name, value in candidate.items() if ".lora_A." in name)
        for name in norms:
            norms[name]["candidate"] = candidate[name].norm().item()
    save_file(candidate, args.output_dir / "adapter_model.safetensors")
    shutil.copyfile(args.zero / "adapter_config.json", args.output_dir / "adapter_config.json")
    fixture = json.loads(args.receipt.read_text())
    if args.heldout:
        fixture["tokens"] = build_heldout_sequences(get_tokenizer(args.model))
    report = {
        "diagnostic_only": True,
        "amplitude_multiplier": 10,
        "minimum_signal_error_ratio": 5,
        "tokens": fixture["tokens"],
        "b_norms": norms,
        "a_unchanged_tensors": sum(".lora_A." in name for name in candidate),
        "trainer_representable": args.trainer_representable,
    }
    for label, path in (
        ("zero", args.zero),
        ("direction", args.direction),
        ("candidate", args.output_dir),
    ):
        with (path / "adapter_model.safetensors").open("rb") as stream:
            report[f"{label}_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        report[name] = run_reference(args.model, args.output_dir, fixture["tokens"], dtype)
    bf16 = torch.tensor(report["bf16"]["updated"], dtype=torch.float64) - torch.tensor(
        report["bf16"]["base"], dtype=torch.float64
    )
    fp32 = torch.tensor(report["fp32"]["updated"], dtype=torch.float64) - torch.tensor(
        report["fp32"]["base"], dtype=torch.float64
    )
    signal = fp32.abs().mean().item()
    error = (bf16 - fp32).abs().mean().item()
    assert error > 0 and torch.isfinite(bf16).all() and torch.isfinite(fp32).all()
    report.update(
        signal_mean_abs=signal,
        arithmetic_error_mean_abs=error,
        signal_error_ratio=signal / error,
    )
    report["signal_separation_passed"] = signal / error >= report["minimum_signal_error_ratio"]
    with (args.output_dir / "reference.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    assert report["signal_separation_passed"], "Predeclared signal/error separation not reached"


if __name__ == "__main__":
    main()
