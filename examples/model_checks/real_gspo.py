"""One-update GSPO primitives for model-enablement checks."""

import math

import torch

from skyrl.tinker.logprob_checks import compare_logprobs


def build_optimizer_batch(batch, trainer_scores, sample_index=0):
    lengths = batch["response_mask"].sum(dim=1).tolist()
    if len(trainer_scores) != sum(lengths):
        raise ValueError("Trainer scores do not align with the batch")
    if sample_index < 0 or sample_index >= batch.batch_size:
        raise ValueError("Optimizer sample index is out of range")
    start = sum(lengths[:sample_index])
    stop = start + lengths[sample_index]
    update = batch[sample_index]
    response_mask = update["response_mask"].bool()
    action_log_probs = torch.zeros_like(update["action_log_probs"])
    action_log_probs[response_mask] = torch.as_tensor(trainer_scores[start:stop], dtype=action_log_probs.dtype)
    advantages = torch.zeros_like(action_log_probs)
    advantages[response_mask] = 1
    update["action_log_probs"] = action_log_probs
    update["advantages"] = advantages
    return update


def validate_rank_receipts(receipts, world_size=8):
    if world_size <= 0 or len(receipts) != world_size or {item["rank"] for item in receipts} != set(range(world_size)):
        raise ValueError(f"Qualification requires exactly trainer ranks 0 through {world_size - 1}")
    return sorted(receipts, key=lambda item: item["rank"])


def run_optimizer_update(policy, batch, world_size=8):
    import ray

    from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput

    results = ray.get(
        policy.async_run_ray_method(
            "mesh",
            "forward_backward",
            data=batch,
            loss_fn="gspo",
            return_per_token_outputs=False,
        )
    )
    output = WorkerOutput.cat(policy.actor_infos, results)
    if not output.metrics or not all(map(math.isfinite, output.metrics.values())):
        raise ValueError("GSPO produced missing or nonfinite metrics")
    gradient_audits = ray.get(policy.async_run_ray_method("pass_through", "describe_lora_gradients"))
    gradient_audits = validate_rank_receipts(gradient_audits, world_size)
    if not all(audit["passed"] for audit in gradient_audits):
        raise ValueError("GSPO produced missing or invalid per-rank LoRA gradients")
    grad_norms = ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
    if len(grad_norms) != world_size or not all(
        norm is not None and math.isfinite(norm) and norm > 0 for norm in grad_norms
    ):
        raise ValueError("Optimizer produced a missing, nonfinite, or nonpositive gradient norm")
    optimizer_audits = ray.get(policy.async_run_ray_method("pass_through", "describe_optimizer_state"))
    optimizer_audits = validate_rank_receipts(optimizer_audits, world_size)
    if not all(audit["passed"] for audit in optimizer_audits):
        raise ValueError("Optimizer produced missing or nonfinite state")
    return {
        "metrics": output.metrics,
        "gradient_audits": gradient_audits,
        "grad_norms": grad_norms,
        "optimizer_audits": optimizer_audits,
    }


def compare_sample(reference, actual, lengths, sample_index):
    if sample_index < 0 or sample_index >= len(lengths):
        raise ValueError("Comparison sample index is out of range")
    start = sum(lengths[:sample_index])
    stop = start + lengths[sample_index]
    return compare_logprobs(reference[start:stop], actual[start:stop])
