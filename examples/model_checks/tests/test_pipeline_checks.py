from copy import deepcopy

import pytest

from examples.model_checks.pipeline_checks import validate_pipeline_coverage


def build_receipts():
    return [
        {
            "rank": stage * 8 + rank,
            "pipeline_rank": stage,
            "tp_rank": rank,
            "layer_numbers": [stage * 2 + 1, stage * 2 + 2],
            "passed": True,
            "coverage": {
                "categories": {
                    "attention": True,
                    "dense_mlp": stage == 0,
                    "routed_expert": True,
                    "shared_expert": True,
                }
            },
        }
        for stage in range(3)
        for rank in range(8)
    ]


def test_pipeline_coverage_accepts_dense_adapters_only_on_the_stage_that_owns_dense_layers():
    receipts = build_receipts()
    assert validate_pipeline_coverage(list(reversed(receipts)), [0, 1, 1, 1, 1, 1], 8, 3) == receipts


@pytest.mark.parametrize("fault", ["missing_stage", "duplicate_stage", "wrong_layers", "missing_shared", "failed"])
def test_pipeline_coverage_rejects_missing_or_misassigned_adapter_evidence(fault):
    receipts = deepcopy(build_receipts())
    if fault == "missing_stage":
        receipts = receipts[:16]
    elif fault == "duplicate_stage":
        receipts[-1]["pipeline_rank"] = 1
    elif fault == "wrong_layers":
        receipts[-1]["layer_numbers"] = [1, 2]
    elif fault == "missing_shared":
        receipts[-1]["coverage"]["categories"]["shared_expert"] = False
    else:
        receipts[-1]["passed"] = False
    with pytest.raises(ValueError):
        validate_pipeline_coverage(receipts, [0, 1, 1, 1, 1, 1], 8, 3)
