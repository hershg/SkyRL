from types import SimpleNamespace

import pytest

from examples.model_checks import run_lora_gspo as runner
from skyrl.tinker.logprob_checks import check_agreement


def test_logprob_budgets_are_inclusive():
    check_agreement({"mean_abs": 0.05, "max_abs": 0.5}, 0.05, 0.5)
    with pytest.raises(AssertionError):
        check_agreement({"mean_abs": 0.050001, "max_abs": 0.5}, 0.05, 0.5)
    with pytest.raises(AssertionError):
        check_agreement({"mean_abs": 0.05, "max_abs": 0.500001}, 0.05, 0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("different_routes", [False, True])
async def test_scores_each_engine_and_rejects_route_disagreement(monkeypatch, different_routes):
    visited = []
    closed = []

    class Client:
        def __init__(self, **kwargs):
            self.url = kwargs["proxy_url"]
            assert kwargs["server_urls"] == [self.url]

        async def aclose(self):
            closed.append(self.url)

    async def score(client, *args):
        visited.append(client.url)
        route = 2 if different_routes and client.url == "engine-b" else 1
        return [[-0.2]], [[[route]]]

    monkeypatch.setattr(runner, "RemoteInferenceClient", Client)
    monkeypatch.setattr(runner, "score_fixed_sampler", score)
    monkeypatch.setattr(runner, "build_routed_batch", lambda *args: "batch")
    monkeypatch.setattr(runner, "score_trainer", lambda *args: [-0.2])
    client = SimpleNamespace(server_urls=["engine-a", "engine-b"], model_name="base", tokenizer=None)
    if different_routes:
        with pytest.raises(ValueError, match="different routes"):
            await runner.score_snapshot(None, client, [[1]], [[2]], "adapter", 0)
    else:
        result = await runner.score_snapshot(None, client, [[1]], [[2]], "adapter", 0)
        assert result["engines"] == {"engine-a": [-0.2], "engine-b": [-0.2]}
        assert result["trainer"] == [-0.2]
    assert visited == closed == ["engine-a", "engine-b"]


@pytest.mark.parametrize(
    "path", ["checkpoint", "/scratch/checkpoint", "/checkpoints", "/checkpoints/../scratch/checkpoint"]
)
def test_distributed_checkpoint_rejects_role_local_or_ambiguous_path(path):
    with pytest.raises(ValueError, match="children of /checkpoints"):
        runner.validate_shared_path(path)
