"""Score a retained adapter independently of Megatron and vLLM."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from skyrl.tinker.logprob_checks import compare_logprobs


@torch.inference_mode()
def score_sequences(model, sequences):
    scores = []
    for sequence in sequences:
        tokens = torch.tensor([sequence], dtype=torch.long, device="cuda")
        logits = model(input_ids=tokens, use_cache=False).logits[0, :-1].float()
        values = logits.log_softmax(-1).gather(-1, tokens[0, 1:, None]).squeeze(-1)
        scores.extend(values.tolist())
    return scores


def run_reference(model_path, adapter_path, sequences, dtype):
    start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="eager")
    model = model.to("cuda").eval()
    base_buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}
    before = score_sequences(model, sequences)
    model = PeftModel.from_pretrained(model, adapter_path, autocast_adapter_dtype=False).eval()
    for name, expected in base_buffers.items():
        actual = model.get_base_model().get_buffer(name)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected), name
    exported = load_file(adapter_path / "adapter_model.safetensors")
    loaded = get_peft_model_state_dict(model)
    assert loaded.keys() == exported.keys()
    for name, tensor in loaded.items():
        assert tensor.dtype == dtype, name
        assert torch.equal(tensor.cpu(), exported[name].to(dtype=dtype)), name
    after = score_sequences(model, sequences)
    with model.disable_adapter():
        disabled = score_sequences(model, sequences)
    result = {
        "base": before,
        "updated": after,
        "disabled": disabled,
        "loaded_adapter_tensors": len(loaded),
        "base_buffer_dtypes": {name: str(buffer.dtype) for name, buffer in base_buffers.items()},
        "seconds": perf_counter() - start,
    }
    result["disabled_parity"] = compare_logprobs(before, disabled)
    assert result["disabled_parity"]["max_abs"] < 1e-6
    result["adapter_change"] = compare_logprobs(before, after)
    del model, loaded, exported, base_buffers
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_reference(reference, receipt):
    comparisons = {}
    delta = torch.tensor(reference["updated"], dtype=torch.float64) - torch.tensor(
        reference["base"], dtype=torch.float64
    )
    for backend, before, after in (
        ("trainer", receipt["trainer_zero"], receipt["trainer_updated"]),
        ("sampler", receipt["zero"], receipt["updated"]),
    ):
        comparisons[f"{backend}_base"] = compare_logprobs(reference["base"], before)
        comparisons[f"{backend}_updated"] = compare_logprobs(reference["updated"], after)
        backend_delta = torch.tensor(after, dtype=torch.float64) - torch.tensor(before, dtype=torch.float64)
        comparisons[f"{backend}_delta"] = compare_logprobs(delta, backend_delta)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    with (args.adapter / "adapter_model.safetensors").open("rb") as adapter:
        digest = hashlib.file_digest(adapter, "sha256").hexdigest()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"diagnostic_only": True, "adapter_sha256": digest, "tokens": receipt["tokens"]}
    for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        reference = run_reference(args.model, args.adapter, receipt["tokens"], dtype)
        report[name] = reference
        reference["comparison"] = compare_reference(reference, receipt)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
