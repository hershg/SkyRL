"""Run one real GSPO update through the LoRA publication lifecycle."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter

import ray
import torch

from examples.model_checks.h1_policy_audit import (
    TARGET_MODULES,
    H1PolicyAuditWorker,
)
from examples.model_checks.megatron_lora import open_runtime, publish, score_trainer
from examples.model_checks.pipeline_checks import validate_pipeline_coverage
from examples.model_checks.real_gspo import (
    build_optimizer_batch,
    compare_sample,
    run_optimizer_update,
    validate_rank_receipts,
)
from examples.model_checks.receiver_checks import (
    fingerprint_receivers,
    validate_receivers,
)
from examples.model_checks.run_lora_logprobs import load_config
from examples.model_checks.tensor_checks import describe_tensor
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.tinker.logprob_checks import (
    build_probe_sequences,
    check_initial_adapter,
    check_updated_adapter,
    check_withheld_publication,
    compare_logprobs,
)
from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)
from skyrl.utils.tok import get_tokenizer

MEAN_ATOL = 0.05
MAX_ATOL = 0.5
MODEL_PATH = "/models/snapshots/304b8051cfb2b260b61ce0cbe330e02a98e73639"


def write_report(output_dir, report):
    temporary = output_dir / "real-gspo.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output_dir / "real-gspo.json")


def validate_shared_path(path):
    path = Path(path)
    if not path.is_absolute() or Path("/checkpoints") not in path.parents or ".." in path.parts:
        raise ValueError("Distributed checkpoint/publication paths must be children of /checkpoints")
    return path


def validate_h1_config(cfg):
    trainer = cfg.trainer
    lora = trainer.policy.model.lora
    validate_shared_path(lora.lora_sync_path)
    parallel = trainer.policy.megatron_config
    engine = cfg.generator.inference_engine
    if trainer.strategy != "megatron" or trainer.placement.colocate_all:
        raise ValueError("H1 requires disaggregated Megatron")
    if (parallel.tensor_model_parallel_size, parallel.expert_model_parallel_size) != (
        8,
        8,
    ):
        raise ValueError("H1 requires trainer TP8/EP8")
    if (parallel.context_parallel_size, parallel.pipeline_model_parallel_size) != (
        1,
        3,
    ):
        raise ValueError("H1 requires trainer CP1/PP3")
    if engine.tensor_parallel_size != 8 or engine.expert_parallel_size != 8 or engine.model_dtype != "bfloat16":
        raise ValueError("H1 requires TP8 BF16 inference")
    if engine.weight_sync_backend != "nccl":
        raise ValueError("H1 requires file LoRA publication")
    if (lora.rank, lora.alpha, lora.max_loras, lora.dtype) != (32, 32, 1, "float32"):
        raise ValueError("H1 requires FP32 rank/alpha-32 single LoRA")
    if tuple(lora.target_modules) != TARGET_MODULES or not lora.share_expert_adapters:
        raise ValueError("H1 LoRA target inventory differs")
    if parallel.lora_config.merge_lora or not lora.lora_sync_path:
        raise ValueError("H1 requires separate shared-file publication")
    if not parallel.moe_enable_routing_replay or not engine.enable_return_routed_experts:
        raise ValueError("H1 requires canonical router replay")
    if trainer.algorithm.temperature != 1 or trainer.algorithm.loss_reduction != "sequence_mean":
        raise ValueError("H1 scoring semantics differ")
    if not trainer.bf16 or trainer.placement.policy_num_nodes != 3 or trainer.placement.policy_num_gpus_per_node != 8:
        raise ValueError("H1 requires three BF16 trainer nodes with exactly 24 ranks")
    if engine.num_engines != 2 or engine.data_parallel_size != 1 or engine.pipeline_parallel_size != 1:
        raise ValueError("H1 requires two independent inference engines")
    kwargs = engine.engine_init_kwargs
    if trainer.seed != 42 or kwargs["seed"] != 42:
        raise ValueError("H1 requires matching fixed initialization seeds")
    if trainer.policy.model.path != MODEL_PATH or kwargs["model"] != MODEL_PATH:
        raise ValueError("H1 requires the frozen GLM-5.3 BF16 revision")
    if kwargs["max_model_len"] != 32768 or kwargs["kv_cache_dtype"] != "bfloat16":
        raise ValueError("H1 requires effective 32K BF16 KV state")
    if kwargs["worker_extension_cls"] != "examples.model_checks.h1_receiver_audit.H1ReceiverAuditWorker":
        raise ValueError("H1 requires its reviewed receiver diagnostic worker")
    if trainer.policy.torch_profiler_config.enable or kwargs.get("profiler_config"):
        raise ValueError("H1 requires disabled profilers")
    if trainer.algorithm.use_kl_loss:
        raise ValueError("H1 requires KL disabled")
    optimizer = trainer.policy.optimizer_config
    if (
        optimizer.lr != 1e-6
        or optimizer.adam_betas != [0.9, 0.999]
        or optimizer.weight_decay != 1e-2
        or optimizer.max_grad_norm != 1.0
        or optimizer.num_warmup_steps != 0
        or optimizer.scheduler != "constant_with_warmup"
        or optimizer.offload_after_step
    ):
        raise ValueError("H1 optimizer semantics differ")
    if parallel.transformer_config_kwargs["moe_router_bias_update_rate"] != 0:
        raise ValueError("H1 requires router-bias update rate zero")


async def generate_responses(client, prompts, model):
    output = await client.generate(
        {
            "prompt_token_ids": prompts,
            "sampling_params": {
                "max_tokens": 64,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": True,
                "seed": 0,
            },
            "session_ids": None,
            "mm_features": None,
            "cache_salt": None,
        },
        model=model,
    )
    return output["response_ids"]


async def score_fixed_sampler(client, prompts, responses, model):
    sequences = [prompt + response for prompt, response in zip(prompts, responses)]
    await client.reset_prefix_cache()
    route_output = await client.generate(
        {
            "prompt_token_ids": sequences,
            "sampling_params": {
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": True,
                "seed": 0,
                "routed_experts_prompt_start": 0,
            },
            "session_ids": None,
            "mm_features": None,
            "cache_salt": None,
        },
        model=model,
    )
    routes = route_output["rollout_expert_indices"]
    if routes is None or any(len(route) != len(sequence) for route, sequence in zip(routes, sequences)):
        raise ValueError("Sampler did not return aligned router replay data")
    scores = []
    for prompt, response, sequence in zip(prompts, responses, sequences):
        output = await client.sample(
            {
                "json": {
                    "model": model,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": sequence}]},
                    "num_samples": 1,
                    "sampling_params": {"max_tokens": 1, "temperature": 0.0, "seed": 0},
                    "prompt_logprobs": True,
                }
            }
        )
        values = output["prompt_logprobs"][len(prompt) :]
        if len(values) != len(response) or not all(value is not None and math.isfinite(value) for value in values):
            raise ValueError("Sampler returned missing or nonfinite fixed-token logprobs")
        scores.append(values)
    return scores, routes


def build_routed_batch(prompts, responses, sampler_scores, routes, pad_token_id):
    masks = [[1] * len(response) for response in responses]
    rewards = [[0.0] * len(response) for response in responses]
    tokens, attention, response_mask, rewards, loss_mask, logprobs, route_tensor = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id,
            prompts,
            responses,
            rewards,
            masks,
            sampler_scores,
            routes,
        )
    )
    if logprobs is None or route_tensor is None:
        raise ValueError("Routed batch is incomplete")
    router_padding = make_router_padding_mask(attention, [len(route) for route in routes])
    zeros = torch.zeros_like(loss_mask)
    batch = TrainingInputBatch(
        {
            "sequences": tokens,
            "attention_mask": attention,
            "response_mask": response_mask,
            "rewards": rewards,
            "loss_mask": loss_mask,
            "rollout_expert_indices": route_tensor,
            "router_padding_mask": router_padding,
            "rollout_logprobs": logprobs,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "advantages": zeros.clone(),
        }
    )
    batch.metadata = {"response_length": response_mask.shape[1]}
    return batch


async def score_snapshot(policy, client, prompts, responses, model, pad_token_id):
    engines = {}
    canonical_routes = None
    snapshot = None
    for url in client.server_urls:
        engine = RemoteInferenceClient(
            proxy_url=url,
            server_urls=[url],
            data_parallel_size=1,
            model_name=client.model_name,
            enable_return_routed_experts=True,
            uses_lora_weight_sync=True,
            tokenizer=client.tokenizer,
        )
        try:
            rows, routes = await score_fixed_sampler(engine, prompts, responses, model)
        finally:
            await engine.aclose()
        route_values = [torch.as_tensor(route).tolist() for route in routes]
        if canonical_routes is not None and route_values != canonical_routes:
            raise ValueError("Inference engines returned different routes for identical scored tokens")
        canonical_routes = route_values
        if snapshot is None:
            batch = build_routed_batch(prompts, responses, rows, routes, pad_token_id)
            snapshot = {
                "batch": batch,
                "trainer": score_trainer(policy, batch),
                "scored_positions": [len(row) for row in rows],
                "routes": routes,
            }
        engines[url] = [score for row in rows for score in row]
    if snapshot is None or len(engines) != 2:
        raise ValueError("Scoring requires exactly two independent inference engines")
    return {**snapshot, "sampler": next(iter(engines.values())), "engines": engines}


def record_snapshot(report, name, snapshot):
    for engine, scores in snapshot["engines"].items():
        evidence = report.setdefault("engines", {}).setdefault(engine, {})
        evidence[name] = scores
        evidence[f"trainer_{name}"] = snapshot["trainer"]
        evidence[f"{name}_scored_positions"] = snapshot["scored_positions"]
    report[name] = snapshot["sampler"]
    report[f"trainer_{name}"] = snapshot["trainer"]
    report[f"{name}_scored_positions"] = snapshot["scored_positions"]
    report[f"{name}_routes"] = [torch.as_tensor(route).tolist() for route in snapshot["routes"]]
    report[f"{name}_contract"] = describe_batch_contract(snapshot["batch"])


def describe_batch_contract(batch):
    fields = (
        "sequences",
        "response_mask",
        "loss_mask",
        "rollout_logprobs",
        "rollout_expert_indices",
        "router_padding_mask",
    )
    result = {}
    for field in fields:
        descriptor = describe_tensor(batch[field])
        descriptor.pop("storage_data_ptr")
        result[field] = descriptor
    targets = describe_tensor(batch["sequences"][:, 1:])
    targets.pop("storage_data_ptr")
    result["target_ids"] = targets
    return result


def get_single_artifact(publications):
    hashes = {item["artifact_sha256"] for item in publications if item["artifact_sha256"] is not None}
    if len(hashes) != 1:
        raise ValueError(f"Expected one published adapter artifact, found {sorted(hashes)}")
    return hashes.pop()


def validate_disk_publications(publications, publication_index, file_path):
    validate_rank_receipts(publications, 24)
    if any(item["publication_index"] != publication_index for item in publications):
        raise ValueError("Trainer publication indices disagree")
    expected_transport = {
        "merge_lora": False,
        "file_path": file_path,
        "colocate_all": False,
    }
    if any(item["transport"] != expected_transport for item in publications):
        raise ValueError("Publication did not use the disaggregated disk path")
    writers = [item for item in publications if item["writer"]]
    if len(writers) != 1 or not writers[0]["tensors"]:
        raise ValueError("Expected one FP32 disk-publication writer")


async def publish_and_audit(policy, client, cfg, publication_index):
    exports = describe_policy(policy, "describe_export_stream")
    if not all(item["passed"] for item in exports):
        raise ValueError("Bridge export stream did not remain FP32")
    await publish(policy, client, cfg)
    receivers = validate_receivers(
        await client.describe_active_lora(), client.server_urls, cfg.generator.inference_engine.tensor_parallel_size
    )
    publications = describe_policy(policy, "describe_disk_publication")
    validate_disk_publications(publications, publication_index, cfg.trainer.policy.model.lora.lora_sync_path)
    loads = publications[0]["receiver_loads"]
    if set(loads) != set(client.server_urls):
        raise ValueError("Publication load responses do not cover both engines")
    for receiver in receivers:
        response = loads[receiver["engine_id"]]
        if response["status"] != 200 or json.loads(response["body"])["lora_int_id"] != receiver["adapter_id"]:
            raise ValueError("Receiver active adapter differs from the published identity")
    evidence = {"artifact_sha256": get_single_artifact(publications)}

    return {"exports": exports, "publications": publications, "receivers": receivers, "route_evidence": evidence}


def describe_policy(policy, method, *args, world_size=24, **kwargs):
    receipts = ray.get(policy.async_run_ray_method("pass_through", method, *args, **kwargs))
    if method == "describe_checkpoint":
        return receipts
    return validate_rank_receipts(receipts, world_size)


def fingerprint_router_bias(audits):
    validate_rank_receipts(audits, 24)
    if any(item["update_rate"] != 0 or not item["tensors"] for item in audits):
        raise ValueError("Live router-bias update rate is not zero")
    return [{name: value["sha256"] for name, value in item["tensors"].items()} for item in audits]


def validate_publication_transition(before, after, changed):
    for field in ("exports", "receivers"):
        if field == "exports":
            first = [{name: value["sha256"] for name, value in rank["tensors"].items()} for rank in before[field]]
            second = [{name: value["sha256"] for name, value in rank["tensors"].items()} for rank in after[field]]
        else:
            first, second = fingerprint_receivers(
                before[field], sorted({item["engine_id"] for item in before[field]}), 8
            ), fingerprint_receivers(after[field], sorted({item["engine_id"] for item in after[field]}), 8)
        if (first != second) != changed:
            raise ValueError(f"Publication {field} do not match the expected update/restore transition")
    if "artifact_sha256" in before["route_evidence"]:
        if (before["route_evidence"]["artifact_sha256"] != after["route_evidence"]["artifact_sha256"]) != changed:
            raise ValueError("File artifact does not match the expected update/restore transition")


def save_restore(policy, checkpoint_dir, world_size=24):
    ray.get(policy.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=str(checkpoint_dir)))
    ray.get(policy.async_run_ray_method("pass_through", "finalize_pending_saves"))
    descriptions = describe_policy(policy, "describe_checkpoint", str(checkpoint_dir), world_size=world_size)
    checkpoint = next((item for item in descriptions if item is not None), None)
    if checkpoint is None or checkpoint["file_count"] == 0 or checkpoint["total_bytes"] == 0:
        raise ValueError("Checkpoint contains no durable data")
    saved = describe_policy(policy, "describe_restorable_state", world_size=world_size)
    mutated = describe_policy(policy, "mutate_restorable_state", world_size=world_size)
    if any(
        before[key] == after[key]
        for before, after in zip(saved, mutated, strict=True)
        for key in ("model", "optimizer", "scheduler")
    ):
        raise ValueError("Checkpoint mutation did not change every rank")
    policy.offload_to_cpu()
    policy.backload_to_gpu()
    ray.get(
        policy.async_run_ray_method(
            "pass_through",
            "load_checkpoint",
            ckpt_dir=str(checkpoint_dir),
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
    )
    restored = describe_policy(policy, "describe_restorable_state", world_size=world_size)
    if restored != saved:
        raise ValueError("Checkpoint did not restore model, optimizer, and scheduler")
    return {
        "files": checkpoint,
        "saved": saved,
        "mutated": mutated,
        "restored": restored,
    }


async def run(args, report):
    if (args.mean_atol, args.max_atol) != (MEAN_ATOL, MAX_ATOL):
        raise ValueError("H1 requires the reviewed inclusive 0.05/0.5 logprob budgets")
    cfg = load_config(args.backend_config, args.output_dir)
    validate_h1_config(cfg)
    validate_shared_path(args.checkpoint_dir)
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    prompts = build_probe_sequences(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    adapter = resolve_policy_model_name(cfg)
    report.update(
        model=cfg.trainer.policy.model.path,
        prompts=prompts,
        mean_atol=args.mean_atol,
        max_atol=args.max_atol,
    )
    async with open_runtime(cfg, tokenizer, H1PolicyAuditWorker, json.loads(args.role_addresses.read_text())) as (
        policy,
        client,
    ):
        report["factors_before"] = describe_policy(policy, "describe_lora_factors")
        pattern = report["factors_before"][0]["moe_layer_pattern"]
        if any(item["moe_layer_pattern"] != pattern for item in report["factors_before"]):
            raise ValueError("Trainer ranks disagree on the model's dense/MoE layout")
        validate_pipeline_coverage(report["factors_before"], pattern, 8, 3)
        if not all(item["passed"] for item in report["factors_before"]):
            raise ValueError("Live FP32 LoRA target inventory is incomplete")
        try:
            report["router_before"] = describe_policy(policy, "describe_router_bias")
            router_fingerprints = fingerprint_router_bias(report["router_before"])
            responses = await generate_responses(client, prompts, client.model_name)
            report["responses"] = responses
            base = await score_snapshot(policy, client, prompts, responses, client.model_name, pad_token_id)
            report["publication_1"] = await publish_and_audit(policy, client, cfg, 1)
            zero = await score_snapshot(policy, client, prompts, responses, adapter, pad_token_id)
            repeat = await score_snapshot(policy, client, prompts, responses, adapter, pad_token_id)
            record_snapshot(report, "base", base)
            record_snapshot(report, "zero", zero)
            record_snapshot(report, "repeat", repeat)
            report["training_contract"] = describe_batch_contract(zero["batch"])
            for evidence in report["engines"].values():
                check_initial_adapter(evidence, args.mean_atol, args.max_atol)
                if evidence["repeat_noise"]["max_abs"] > 1e-6:
                    raise ValueError("Inference engine is not repeatable")
            check_initial_adapter(report, args.mean_atol, args.max_atol)
            if report["repeat_noise"]["max_abs"] > 1e-6 or report["trainer_repeat_noise"]["max_abs"] > 1e-6:
                raise ValueError("Zero-init policy is not repeatable")

            update_batch = build_optimizer_batch(zero["batch"], zero["trainer"])
            report["optimizer"] = run_optimizer_update(policy, update_batch, world_size=24)
            report["factors_after_update"] = describe_policy(policy, "describe_lora_factors")
            if not all(item["passed"] for item in report["factors_after_update"]):
                raise ValueError("FP32 LoRA factor audit failed after optimizer update")
            report["router_after"] = describe_policy(policy, "describe_router_bias")
            if fingerprint_router_bias(report["router_after"]) != router_fingerprints:
                raise ValueError("Router-bias tensors changed during the optimizer update")
            stale = await score_snapshot(policy, client, prompts, responses, adapter, pad_token_id)
            record_snapshot(report, "stale", stale)
            report["trainer_updated"] = stale["trainer"]
            for evidence in report["engines"].values():
                check_withheld_publication(evidence)
                if evidence["withheld_publication"]["max_abs"] > 1e-6:
                    raise ValueError("Inference engine changed before publication")
            check_withheld_publication(report)
            if report["withheld_publication"]["max_abs"] > 1e-6:
                raise ValueError("Sampler changed while publication was withheld")
            report["trainer_change"] = compare_logprobs(report["trainer_zero"], report["trainer_updated"])
            if report["trainer_change"]["mean_abs"] <= 1e-6:
                raise ValueError("Real optimizer update did not measurably change trainer scores")

            report["publication_2"] = await publish_and_audit(policy, client, cfg, 2)
            validate_publication_transition(report["publication_1"], report["publication_2"], changed=True)
            updated = await score_snapshot(policy, client, prompts, responses, adapter, pad_token_id)
            record_snapshot(report, "updated", updated)
            for evidence in report["engines"].values():
                check_updated_adapter(evidence, args.mean_atol, args.max_atol)
                heldout = compare_sample(
                    evidence["trainer_updated"], evidence["updated"], updated["scored_positions"], 1
                )
                evidence["heldout_updated_parity"] = heldout
                if heldout["mean_abs"] > args.mean_atol or heldout["max_abs"] > args.max_atol:
                    raise ValueError("Inference engine held-out parity exceeded its budget")
            check_updated_adapter(report, args.mean_atol, args.max_atol)
            report["heldout_updated_parity"] = compare_sample(
                updated["trainer"], updated["sampler"], updated["scored_positions"], 1
            )
            if (
                report["heldout_updated_parity"]["mean_abs"] > args.mean_atol
                or report["heldout_updated_parity"]["max_abs"] > args.max_atol
            ):
                raise ValueError("Held-out updated parity exceeded its budget")

            report["checkpoint"] = save_restore(policy, args.checkpoint_dir)
            restored_trainer = score_trainer(policy, updated["batch"])
            report["restored_trainer_change"] = compare_logprobs(updated["trainer"], restored_trainer)
            if report["restored_trainer_change"]["max_abs"] > 1e-6:
                raise ValueError("Checkpoint restore changed trainer logprobs")
            report["publication_3"] = await publish_and_audit(policy, client, cfg, 3)
            validate_publication_transition(report["publication_2"], report["publication_3"], changed=False)
            restored = await score_snapshot(policy, client, prompts, responses, adapter, pad_token_id)
            record_snapshot(report, "restored", restored)
            for evidence in report["engines"].values():
                parity = compare_logprobs(evidence["trainer_restored"], evidence["restored"])
                evidence["restored_parity"] = parity
                if parity["mean_abs"] > args.mean_atol or parity["max_abs"] > args.max_atol:
                    raise ValueError("Inference engine restored parity exceeded its budget")
            report["restored_parity"] = compare_logprobs(restored["trainer"], restored["sampler"])
            if (
                report["restored_parity"]["mean_abs"] > args.mean_atol
                or report["restored_parity"]["max_abs"] > args.max_atol
            ):
                raise ValueError("Restored publication parity exceeded its budget")
            await client.unload_lora_adapter(adapter)
            report["receiver_eviction"] = await client.evict_active_lora()
            expected = {(engine, rank) for engine in client.server_urls for rank in range(8)}
            observed = {(item["engine_id"], item["tp_rank"]) for item in report["receiver_eviction"]}
            if len(report["receiver_eviction"]) != 16 or observed != expected:
                raise ValueError("Adapter eviction did not cover all receiver ranks")
            report["adapter_unloaded"] = True
            policy.offload_to_cpu()
            report["trainer_offloaded"] = True
        finally:
            write_report(args.output_dir, report)
    report["runtime_released"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-addresses", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-atol", type=float, choices=[MEAN_ATOL], default=MEAN_ATOL)
    parser.add_argument("--max-atol", type=float, choices=[MAX_ATOL], default=MAX_ATOL)
    args = parser.parse_args()
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
