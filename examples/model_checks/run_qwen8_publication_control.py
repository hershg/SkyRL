"""Diagnostic-only, predeclared Qwen8 B32 control on two fixed token fixtures."""

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import ray
import torch
from safetensors.torch import load_file, save_file

from examples.model_checks.heldout_publication import (
    build_heldout_sequences,
    negate_adapter_b,
)
from examples.model_checks.megatron_lora import (
    build_batch,
    open_runtime,
    publish,
    score_sampler,
    score_trainer,
)
from examples.model_checks.run_active_lora_audit import audit
from examples.tinker.glm53.run_lora_logprobs import (
    check_unpublished_sampler,
    check_zero_initialized_policy,
    load_config,
)
from skyrl.backends.skyrl_train.inference_servers.utils import (
    build_vllm_cli_args,
    resolve_policy_model_name,
)
from skyrl.tinker.logprob_checks import build_probe_sequences, compare_logprobs
from skyrl.utils.tok import get_tokenizer


def write_report(path, report):
    pending = path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False))
    pending.replace(path)


def summarize_control(report):
    controls = {
        phase: compare_logprobs(report["trainer_updated"], report[phase])
        for phase in ("stale", "updated", "wrong", "restored")
    }
    trainer_delta = torch.tensor(report["trainer_updated"], dtype=torch.float64) - torch.tensor(
        report["trainer_zero"], dtype=torch.float64
    )
    sampler_delta = torch.tensor(report["updated"], dtype=torch.float64) - torch.tensor(
        report["zero"], dtype=torch.float64
    )
    delta = compare_logprobs(trainer_delta, sampler_delta)
    delta.update(
        cosine=torch.nn.functional.cosine_similarity(trainer_delta, sampler_delta, dim=0).item(),
        scale=(sampler_delta.norm() / trainer_delta.norm()).item(),
        relative_l2=((sampler_delta - trainer_delta).norm() / trainer_delta.norm()).item(),
    )
    report.update(
        publication_controls=controls,
        publication_controls_passed=(
            controls["stale"]["mean_abs"] >= 0.05
            and controls["wrong"]["mean_abs"] >= 0.05
            and controls["updated"]["mean_abs"] < 0.05
            and controls["restored"]["mean_abs"] < 0.05
        ),
        strict_delta_passed=delta["mean_abs"] < 0.005,
        update_delta=delta,
        trainer_unchanged=compare_logprobs(report["trainer_updated"], report["trainer_after_restore"]),
        restore_repeat=compare_logprobs(report["updated"], report["restored"]),
    )


async def run(args, report):
    cfg = load_config(args.backend_config, args.output_dir)
    extension = "examples.model_checks.active_lora_worker.ActiveLoRAAuditWorker"
    cfg.generator.inference_engine.engine_init_kwargs["worker_extension_cls"] = extension
    engine = build_vllm_cli_args(cfg)
    assert engine.worker_extension_cls == extension and engine.tensor_parallel_size == 1
    report["resolved_inference"] = {
        key: getattr(engine, key)
        for key in (
            "worker_extension_cls",
            "enforce_eager",
            "dtype",
            "kv_cache_dtype",
            "max_model_len",
        )
    }
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    sequences = {
        "original": build_probe_sequences(tokenizer),
        "heldout": build_heldout_sequences(tokenizer),
    }
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batches = {name: build_batch(tokens, pad_id) for name, tokens in sequences.items()}
    report["fixtures"] = {name: {"tokens": tokens} for name, tokens in sequences.items()}
    adapter = resolve_policy_model_name(cfg)
    exports = Path(cfg.trainer.policy.model.lora.lora_sync_path)
    async with open_runtime(cfg, tokenizer) as (policy, client):
        try:
            for name, tokens in sequences.items():
                await check_zero_initialized_policy(
                    policy,
                    client,
                    cfg,
                    batches[name],
                    tokens,
                    report["fixtures"][name],
                    0.05,
                )
            zero = args.output_dir / "zero-export"
            shutil.copytree(exports, zero)
            report["zero_buffers"] = await audit(client, zero, report, "zero")
            assert report["zero_buffers"]["passed"]
            report["perturbation"] = ray.get(policy.async_run_ray_method("pass_through", "perturb_test_b_only", 32))
            for name, tokens in sequences.items():
                fixture = report["fixtures"][name]
                fixture["trainer_updated"] = score_trainer(policy, batches[name])
                await check_unpublished_sampler(client, tokens, adapter, fixture)
            await publish(policy, client, cfg)
            updated = args.output_dir / "updated-export"
            shutil.copytree(exports, updated)
            original = load_file(zero / "adapter_model.safetensors")
            changed = load_file(updated / "adapter_model.safetensors")
            assert original.keys() == changed.keys()
            assert all(torch.equal(tensor, changed[name]) for name, tensor in original.items() if ".lora_A." in name)
            report["b_norms"] = {name: tensor.norm().item() for name, tensor in changed.items() if ".lora_B." in name}
            with (updated / "adapter_model.safetensors").open("rb") as stream:
                report["adapter_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            write_report(
                args.output_dir / "reference-ready.json",
                {"tokens": sequences, "adapter_sha256": report["adapter_sha256"]},
            )
            report["updated_buffers"] = await audit(client, updated, report, "updated")
            report["stale_export_negative"] = await audit(client, zero, report, "stale")
            assert report["updated_buffers"]["passed"] and not report["stale_export_negative"]["passed"]
            for name, tokens in sequences.items():
                report["fixtures"][name]["updated"] = await score_sampler(client, tokens, adapter)
            wrong = args.output_dir / "wrong-export"
            shutil.copytree(updated, wrong)
            save_file(negate_adapter_b(changed), wrong / "adapter_model.safetensors")
            await client.load_lora_adapter(adapter, str(wrong))
            report["wrong_buffers"] = await audit(client, wrong, report, "wrong")
            assert report["wrong_buffers"]["passed"]
            for name, tokens in sequences.items():
                report["fixtures"][name]["wrong"] = await score_sampler(client, tokens, adapter)
            await client.load_lora_adapter(adapter, str(updated))
            report["restored_buffers"] = await audit(client, updated, report, "restored")
            assert report["restored_buffers"]["passed"]
            for name, tokens in sequences.items():
                fixture = report["fixtures"][name]
                fixture["restored"] = await score_sampler(client, tokens, adapter)
                fixture["trainer_after_restore"] = score_trainer(policy, batches[name])
                summarize_control(fixture)
            report["completed"] = True
        finally:
            write_report(args.output_dir / "logprobs.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "diagnostic_only": True,
        "completed": False,
        "b_multiplier": 32,
        "mean_atol": 0.05,
        "delta_mean_atol": 0.005,
    }
    start = perf_counter()
    try:
        asyncio.run(run(args, report))
    finally:
        report["seconds"] = perf_counter() - start
        write_report(args.output_dir / "logprobs.json", report)


if __name__ == "__main__":
    main()
