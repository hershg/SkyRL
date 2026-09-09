"""Diagnose native LoRA score sensitivity to trainer batching in one warm runtime."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import torch

from examples.model_checks.megatron_lora import (
    build_batch,
    open_runtime,
    publish,
    score_sampler,
    score_trainer,
)
from examples.tinker.glm53.run_lora_logprobs import (
    apply_trainer_update,
    check_unpublished_sampler,
    check_zero_initialized_policy,
    load_config,
)
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args, resolve_policy_model_name
from skyrl.tinker.logprob_checks import (
    build_probe_sequences,
    check_updated_adapter,
    compare_logprobs,
)
from skyrl.utils.tok import get_tokenizer


def score_individual_datums(policy, batches):
    return [score for batch in batches for score in score_trainer(policy, batch)]


def compare_update_deltas(trainer_before, trainer_after, sampler_before, sampler_after):
    trainer_delta = torch.tensor(trainer_after, dtype=torch.float64) - torch.tensor(trainer_before, dtype=torch.float64)
    sampler_delta = torch.tensor(sampler_after, dtype=torch.float64) - torch.tensor(sampler_before, dtype=torch.float64)
    return compare_logprobs(trainer_delta, sampler_delta)


async def run(args, report):
    cfg = load_config(args.backend_config, args.output_dir)
    # SkyRL resets eager LoRA to compiled for speed; this is an explicit numerical control.
    cfg.generator.inference_engine.enforce_eager = True
    engine_args = build_vllm_cli_args(cfg)
    assert engine_args.enforce_eager is True
    report["effective_inference"] = {
        "enforce_eager": engine_args.enforce_eager,
        "dtype": engine_args.dtype,
        "kv_cache_dtype": engine_args.kv_cache_dtype,
        "max_model_len": engine_args.max_model_len,
    }
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    sequences = build_probe_sequences(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_id)
    individual_batches = [build_batch([tokens], pad_id) for tokens in sequences]
    report.update(
        tokens=sequences,
        scored_positions=[len(tokens) - 1 for tokens in sequences],
        mean_atol=0.05,
        delta_mean_atol=0.005,
        model=cfg.trainer.policy.model.path,
    )
    adapter = resolve_policy_model_name(cfg)
    async with open_runtime(cfg, tokenizer) as (policy, client):
        await check_zero_initialized_policy(policy, client, cfg, batch, sequences, report, 0.05)
        report["trainer_zero_individual"] = score_individual_datums(policy, individual_batches)
        apply_trainer_update(policy, batch, report)
        report["trainer_updated_individual"] = score_individual_datums(policy, individual_batches)
        await check_unpublished_sampler(client, sequences, adapter, report)
        await publish(policy, client, cfg)
        report["updated"] = await score_sampler(client, sequences, adapter)

        for phase in ("zero", "updated"):
            report[f"trainer_{phase}_batching"] = compare_logprobs(
                report[f"trainer_{phase}"], report[f"trainer_{phase}_individual"]
            )
            report[f"individual_{phase}_parity"] = compare_logprobs(
                report[f"trainer_{phase}_individual"], report[phase]
            )
        report["individual_update_delta"] = compare_update_deltas(
            report["trainer_zero_individual"],
            report["trainer_updated_individual"],
            report["zero"],
            report["updated"],
        )
        check_updated_adapter(report, 0.05, 0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "diagnostic_only": True}
    start = perf_counter()
    try:
        asyncio.run(run(args, report))
        report["passed"] = True
    finally:
        report["seconds"] = perf_counter() - start
        with (args.output_dir / "logprobs.json").open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
