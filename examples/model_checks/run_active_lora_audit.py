"""Check exact Qwen TP1 adapter publication, preserving the native numerical checks."""

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from time import perf_counter

import httpx
import ray
import torch
from safetensors.torch import load_file

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
from skyrl.backends.skyrl_train.inference_servers.utils import (
    build_vllm_cli_args,
    resolve_policy_model_name,
)
from skyrl.tinker.logprob_checks import build_probe_sequences, check_updated_adapter
from skyrl.utils.tok import get_tokenizer


async def call_worker(client, method, args):
    assert len(client.server_urls) == 1
    async with httpx.AsyncClient(timeout=120) as http:
        response = await http.post(
            f"{client.server_urls[0]}/collective_rpc",
            json={"method": method, "args": args, "timeout": 90},
        )
        response.raise_for_status()
        results = response.json()["results"]
    assert len(results) == 1, results
    return results[0]


async def audit(client, path, report, phase):
    report[f"{phase}_layout"] = await call_worker(client, "describe_active_lora", [])
    with (path.parent / f"{phase}-buffer-layout.json").open("x") as output:
        json.dump(report[f"{phase}_layout"], output, indent=2, allow_nan=False)
    return await call_worker(client, "audit_active_lora", [str(path)])


async def run(args, report):
    cfg = load_config(args.backend_config, args.output_dir)
    extension = "examples.model_checks.active_lora_worker.ActiveLoRAAuditWorker"
    cfg.generator.inference_engine.engine_init_kwargs["worker_extension_cls"] = extension
    engine = build_vllm_cli_args(cfg)
    assert engine.worker_extension_cls == extension and engine.tensor_parallel_size == 1
    report["resolved_inference"] = {
        key: getattr(engine, key)
        for key in ("worker_extension_cls", "enforce_eager", "dtype", "kv_cache_dtype", "max_model_len")
    }
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    sequences = build_probe_sequences(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_id)
    adapter = resolve_policy_model_name(cfg)
    report.update(tokens=sequences, mean_atol=0.05, delta_mean_atol=0.005, model=cfg.trainer.policy.model.path)
    exports = Path(cfg.trainer.policy.model.lora.lora_sync_path)
    async with open_runtime(cfg, tokenizer) as (policy, client):
        await check_zero_initialized_policy(policy, client, cfg, batch, sequences, report, 0.05)
        zero = args.output_dir / "zero-export"
        shutil.copytree(exports, zero)
        report["zero_buffers"] = await audit(client, zero, report, "zero")
        assert report["zero_buffers"]["passed"], report["zero_buffers"]
        if args.b_only_candidate is None:
            apply_trainer_update(policy, batch, report)
        else:
            report["perturbation"] = ray.get(policy.async_run_ray_method("pass_through", "perturb_test_b_only"))
            report["trainer_updated"] = score_trainer(policy, batch)
        await check_unpublished_sampler(client, sequences, adapter, report)
        await publish(policy, client, cfg)
        report["updated"] = await score_sampler(client, sequences, adapter)
        updated = args.output_dir / "updated-export"
        shutil.copytree(exports, updated)
        if args.b_only_candidate is not None:
            candidate = load_file(args.b_only_candidate / "adapter_model.safetensors")
            actual = load_file(updated / "adapter_model.safetensors")
            assert candidate.keys() == actual.keys()
            report["candidate_mismatched_tensors"] = [
                name
                for name, tensor in actual.items()
                if not torch.equal(tensor, candidate[name].to(torch.bfloat16).float())
            ]
            assert not report["candidate_mismatched_tensors"], report["candidate_mismatched_tensors"]
            report["exact_representable_candidate_tensors"] = len(actual)
        report["updated_buffers"] = await audit(client, updated, report, "updated")
        report["stale_export_negative"] = await audit(client, zero, report, "stale")
        assert report["updated_buffers"]["passed"], report["updated_buffers"]
        assert not report["stale_export_negative"]["passed"], "Stale export was not detected"
        report["tensor_integrity_passed"] = True
        check_updated_adapter(report, 0.05, 0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b-only-candidate", type=Path, help="Diagnostic-only retained B-only10x reference export")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "diagnostic_only": True, "tensor_integrity_passed": False}
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
