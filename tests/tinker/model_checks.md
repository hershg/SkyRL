# LoRA runtime checks before long training jobs

These helpers exercise an **already-owned SkyRL endpoint** through the Tinker SDK.
They do not deploy services, start Ray, or require hosted Tinker/TCLI.

## Two checks

| Helper | Coverage |
| --- | --- |
| `check_lora_runtime` | Mixed-length example ordering; trainer/sampler scores on identical tokens before and after a real PPO update; updated-adapter publication. |
| `check_lora_capacity` | Exact-context input/scored positions; two accumulated backwards per optimizer, repeated for two optimizer/publication/sample cycles; final checkpoint. |

Both check the server-reported LoRA rank. Full-weight clients and wrong-rank adapters
are rejected. These checks do not replace existing full-weight, multi-adapter isolation
or backend/kernel tests.

## Calling them

Import from `tests.tinker.model_checks` in a pinned SkyRL checkout.
The existing integration runner owns the client, deadline, report persistence and cleanup:

```python
trainer = service.create_lora_training_client(base_model=model_name, rank=rank, seed=0)
report = {}
try:
    check_lora_runtime(
        trainer, rank, short_datums, advantages, adam,
        mean_atol=approved_mean_atol, update_atol=approved_update_atol,
        batching_atol=approved_batching_atol, report=report,
    )
finally:
    persist_report(report)  # Use the runner's artifact writer.
```

Supply fixed, contiguous shifted tokens, CE response weights and mixed-length examples.
The helper builds one PPO update from pre-update trainer scores and caller advantages.
Use the same reviewed examples/settings as the reference test; no random adapter noise.
The server's rendered optimizer configuration must match `adam`: some optimizer fields
are fixed at model creation and cannot be changed by the step request.

Tolerances are **required, model/fixture-specific inputs**, not GLM defaults established
by this PR. CPU fake-model values are not live acceptance budgets. Review budgets using
a known-good fixture and stale-publication negative control before enabling a CI gate.
Both absolute and update differences are checked. An unresolved update fails as
inconclusive; do not force a larger update merely to obtain a pass.

The batching check compares mixed-length batched trainer output to separately scored
examples in the original order. It does not claim bitwise inference-batch invariance.

## Full-context qualification

Call `check_lora_capacity` separately with two saved backward batches, the advertised
`context_length`, and the **actual production loss/config** (`ppo` or `cross_entropy`).
Supply Adam parameters, a `context_length - 1` sampling prompt and a unique checkpoint name.

- Include an intact exact-context datum with every input position scored and representative
  heterogeneous packed examples. For 256K this means 262,144 positions, not 262,143.
- PPO fixtures need valid reference logprobs and nonzero advantages on the exact-length datum.
  A CE memory result must not be reported as qualification of a PPO recipe.
- Keep the same model throughout: backward A → backward B → optimizer → publication →
  sample, twice. This exercises retained gradients and the next cycle's optimizer state.
- Each one-token sample reaches the declared total context. Returned training logprobs and
  available numeric metrics must be finite. Save the training checkpoint after both cycles.

This is memory/lifecycle coverage of the supplied fixtures, not proof that every packing
shape fits or that long-context trainer/sampler numerical parity is established.
Short numerical agreement does not automatically cover context-dependent kernels.
The caller retains separate restore/cold-replay checks; saving a checkpoint path alone
does not establish durable reloadability. Record rendered packing and sequence limits.

## Where it fits

```text
deploy → LoRA runtime check → opt-in capacity check → existing fast/medium/long jobs
```

SkyRL owns these shared checks. Trajectory should invoke them from its existing regression
DAG and own service claims, teardown, artifacts, real-task rollout accounting and learning
evals. Do not duplicate thresholds or add a second scheduler/framework.

The small check uses the **ephemeral RL publication API**. Unlike the earlier gist's named
persistent sampler checkpoint, it does not request a durable sampler save; do not directly
compare those timings. A rollout-load/queue-admission test is separate from numerical
batching correctness and can stay in the existing workload/load suite.

Record model/tokenizer/source/image pins, target modules, precision, topology, fixture
tokens and phase times alongside each report. Collect GPU memory/restart telemetry from
the runtime. A client timeout does not prove remote cancellation: drain before reuse.

## Validation

CPU controls: `uv run --isolated --extra tinker --extra dev pytest tests/tinker/test_model_checks.py`.
They cover stale adapters, wrong rank/full weights, batch corruption, NaNs, off-by-one
context and OOMs on later backwards. No live GPU or GLM qualification is claimed.
