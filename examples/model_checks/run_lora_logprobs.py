"""Check native LoRA publication on an owned Ray cluster."""

import argparse
import asyncio
import json
import math
from contextlib import AsyncExitStack
from pathlib import Path
from time import perf_counter

from examples.model_checks.megatron_lora import (
    build_batch,
    open_runtime,
    perturb_trainer,
    publish,
    score_sampler,
    score_trainer,
)
from examples.model_checks.paired_completion import score_with_routes
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.tinker.logprob_checks import build_probe_sequences as build_sequences
from skyrl.tinker.logprob_checks import (
    check_initial_adapter,
    check_update_stimulus,
    check_updated_adapter,
    check_withheld_publication,
    compare_logprobs,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer


async def run(args, report):
    cfg = load_config(args.backend_config, args.output_dir)
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    sequences = build_sequences(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_token_id)
    report.update(
        tokens=sequences,
        scored_positions=[len(tokens) - 1 for tokens in sequences],
        mean_atol=args.mean_atol,
        max_atol=args.max_atol,
        lora_b_multiplier=args.lora_b_multiplier,
        model=cfg.trainer.policy.model.path,
    )
    adapter = resolve_policy_model_name(cfg)

    async with open_runtime(cfg, tokenizer) as (policy, client):
        try:
            replay = cfg.trainer.policy.megatron_config.moe_enable_routing_replay
            if replay:
                await check_all_replayed_engines(policy, client, cfg, batch, sequences, pad_token_id, report, args)
                return
            await check_zero_initialized_policy(
                policy,
                client,
                cfg,
                batch,
                sequences,
                report,
                args.mean_atol,
                args.max_atol,
            )
            apply_trainer_update(policy, batch, report, args.lora_b_multiplier)
            await check_unpublished_sampler(client, sequences, adapter, report)
            check_update_stimulus(report, args.mean_atol)
            await publish(policy, client, cfg)
            await check_published_update(client, sequences, adapter, report, args.mean_atol, args.max_atol)
        finally:
            # Preserve failed assertions even if subsequent runtime cleanup hangs.
            write_report(args.output_dir, report)


async def check_all_replayed_engines(policy, client, cfg, batch, sequences, pad_token_id, report, args):
    count = cfg.generator.inference_engine.num_engines
    if count == 1:
        await check_replayed_policy(policy, client, cfg, batch, sequences, pad_token_id, report, args)
        return
    urls = client.server_urls
    if len(urls) != count or len(set(urls)) != count:
        raise ValueError("Expected one distinct server URL per inference engine")
    report["engines"] = [{"server_url": url, "passed": False} for url in urls]
    async with AsyncExitStack() as stack:
        targets = []
        for url, engine_report in zip(urls, report["engines"], strict=True):
            direct = RemoteInferenceClient(
                proxy_url=url,
                server_urls=[url],
                data_parallel_size=1,
                model_name=client.model_name,
                enable_return_routed_experts=client.enable_return_routed_experts,
                uses_lora_weight_sync=client.uses_lora_weight_sync,
                tokenizer=client.tokenizer,
            )
            stack.push_async_callback(direct.aclose)
            targets.append((direct, engine_report))
        await check_replayed_policy(policy, client, cfg, batch, sequences, pad_token_id, report, args, targets)


async def check_replayed_policy(
    policy, client, cfg, unreplayed_batch, sequences, pad_token_id, report, args, targets=None
):
    targets = [(client, report)] if targets is None else targets

    async def score_phase(target, result, phase, model):
        scores, routes = await score_with_routes(target, sequences, model)
        result[phase] = scores
        result[f"{phase}_routes"] = [route.tolist() for route in routes]
        return build_batch(sequences, pad_token_id, routes)

    for target, result in targets:
        await score_phase(target, result, "base", client.model_name)
    await publish(policy, client, cfg)
    adapter = resolve_policy_model_name(cfg)
    zero_batches = []
    for target, result in targets:
        zero_batch = await score_phase(target, result, "zero", adapter)
        zero_batches.append(zero_batch)
        result["trainer_zero_unreplayed"] = score_trainer(policy, unreplayed_batch)
        result["zero_parity_unreplayed"] = compare_logprobs(result["trainer_zero_unreplayed"], result["zero"])
        result["trainer_zero"] = score_trainer(policy, zero_batch)
        repeat_batch = await score_phase(target, result, "repeat", adapter)
        result["trainer_repeat"] = score_trainer(policy, repeat_batch)
        result["repeat_noise"] = compare_logprobs(result["zero"], result["repeat"])
        result["trainer_repeat_noise"] = compare_logprobs(result["trainer_zero"], result["trainer_repeat"])
        for field in ("repeat_noise", "trainer_repeat_noise"):
            if result[field]["max_abs"] > 1e-6:
                raise AssertionError(f"{field} exceeds 1e-6: {result[field]}")
        check_initial_adapter(result, args.mean_atol, args.max_atol)

    perturbation = perturb_trainer(policy, args.lora_b_multiplier)
    for (target, result), zero_batch in zip(targets, zero_batches, strict=True):
        result["perturbation"] = perturbation
        result["trainer_updated"] = score_trainer(policy, zero_batch)
        result["trainer_updated_before_publication"] = result["trainer_updated"]
        await score_phase(target, result, "stale", adapter)
        check_withheld_publication(result)
        check_update_stimulus(result, args.mean_atol)
        result["stale_parity_before_publication"] = result["stale_parity"]
    await publish(policy, client, cfg)
    for target, result in targets:
        updated_batch = await score_phase(target, result, "updated", adapter)
        result["trainer_updated"] = score_trainer(policy, updated_batch)
        check_updated_adapter(result, args.mean_atol, args.max_atol)
        await score_phase(target, result, "updated_repeat", adapter)
        check_updated_repeat(result)
        if result is not report:
            result["passed"] = True


def write_report(output_dir, report):
    temporary = output_dir / "logprobs.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output_dir / "logprobs.json")


async def check_zero_initialized_policy(policy, client, cfg, batch, sequences, report, mean_atol, max_atol):
    report["base"] = await score_sampler(client, sequences, client.model_name)
    await publish(policy, client, cfg)
    adapter = resolve_policy_model_name(cfg)
    report["zero"] = await score_sampler(client, sequences, adapter)
    report["trainer_zero"] = score_trainer(policy, batch)
    report["repeat"] = await score_sampler(client, sequences, adapter)
    report["trainer_repeat"] = score_trainer(policy, batch)
    check_initial_adapter(report, mean_atol, max_atol)


def apply_trainer_update(policy, batch, report, multiplier=10):
    report["perturbation"] = perturb_trainer(policy, multiplier)
    report["trainer_updated"] = score_trainer(policy, batch)


async def check_unpublished_sampler(client, sequences, adapter, report):
    report["stale"] = await score_sampler(client, sequences, adapter)
    check_withheld_publication(report)


async def check_published_update(client, sequences, adapter, report, mean_atol, max_atol):
    report["updated"] = await score_sampler(client, sequences, adapter)
    check_updated_adapter(report, mean_atol, max_atol)
    report["updated_repeat"] = await score_sampler(client, sequences, adapter)
    check_updated_repeat(report)


def check_updated_repeat(report):
    report["updated_repeat_noise"] = compare_logprobs(report["updated"], report["updated_repeat"])
    if report["updated_repeat_noise"]["max_abs"] > 1e-6:
        raise AssertionError(f"updated_repeat_noise exceeds 1e-6: {report['updated_repeat_noise']}")


def validate_config(overrides):
    replay = overrides.get("trainer.policy.megatron_config.moe_enable_routing_replay", False)
    capture = overrides.get("generator.inference_engine.enable_return_routed_experts", False)
    if replay != capture:
        raise ValueError("Route replay and inference route capture must be enabled together")
    if overrides.get("generator.inference_engine.num_engines", 1) > 1:
        if not replay or overrides.get("generator.inference_engine.data_parallel_size", 1) != 1:
            raise ValueError("Multi-engine checks require route replay and inference DP=1")
    if overrides["strategy"] != "megatron":
        raise ValueError("This diagnostic requires Megatron")
    if overrides["trainer.placement.colocate_all"]:
        raise ValueError("This diagnostic requires disaggregated trainer/inference GPUs")
    if overrides["trainer.policy.model.lora.rank"] <= 0:
        raise ValueError("A positive LoRA rank is required")
    if overrides["trainer.policy.megatron_config.lora_config.merge_lora"]:
        raise ValueError("Separate adapter publication is required")
    if not overrides["generator.inference_engine.run_engines_locally"]:
        raise ValueError("This command starts its own inference engines")
    for key in (
        "generator.inference_engine.external_proxy_url",
        "generator.inference_engine.external_server_urls",
        "generator.inference_engine.enable_pd",
    ):
        if overrides.get(key):
            raise ValueError(f"This owned-runtime diagnostic does not support {key}")


def validate_scoring_config(trainer):
    if trainer.algorithm.temperature != 1.0:
        raise ValueError("Prompt-logprob comparison requires trainer temperature=1")
    placement = trainer.placement
    world_size = placement.policy_num_nodes * placement.policy_num_gpus_per_node
    parallel = trainer.policy.megatron_config
    model_size = (
        parallel.tensor_model_parallel_size * parallel.pipeline_model_parallel_size * parallel.context_parallel_size
    )
    if world_size <= 0 or model_size <= 0 or world_size % model_size:
        raise ValueError("Trainer GPU count must be divisible by TP * PP * CP")
    if world_size // model_size not in (1, 2):
        raise ValueError("The two probe sequences require trainer DP=1 or DP=2")


def load_config(path, output_dir):
    overrides = json.loads(path.read_text())
    validate_config(overrides)
    overrides["trainer.strategy"] = overrides.pop("strategy")
    # Disable measurement overhead; preserve model, topology and kernel settings.
    overrides["trainer.policy.torch_profiler_config"] = {"enable": False}
    overrides["trainer.log_path"] = str(output_dir / "runtime-logs")
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    validate_scoring_config(cfg.trainer)
    with (output_dir / "backend-config.json").open("x") as receipt:
        json.dump(overrides, receipt, indent=2)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-config",
        type=Path,
        required=True,
        help="Rendered run_server.py backend config",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mean-atol",
        type=float,
        required=True,
        help="Reviewed mean logprob error budget",
    )
    parser.add_argument(
        "--max-atol",
        type=float,
        required=True,
        help="Reviewed maximum token logprob error budget",
    )
    parser.add_argument(
        "--lora-b-multiplier",
        type=float,
        default=10,
        help="Predeclared test stimulus, not an adaptive acceptance knob",
    )
    args = parser.parse_args()
    if any(not math.isfinite(bound) or bound <= 0 for bound in (args.mean_atol, args.max_atol)):
        parser.error("logprob budgets must be positive and finite")
    if not math.isfinite(args.lora_b_multiplier) or args.lora_b_multiplier <= 0:
        parser.error("LoRA B multiplier must be positive and finite")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False}
    started = perf_counter()
    try:
        asyncio.run(run(args, report))
        report["passed"] = True
    finally:
        report["seconds"] = perf_counter() - started
        write_report(args.output_dir, report)


if __name__ == "__main__":
    main()
