"""Check complete layer and adapter coverage across pipeline stages."""

from examples.model_checks.real_gspo import validate_rank_receipts


def validate_pipeline_coverage(receipts, moe_layer_pattern, tp_size, pp_size):
    if tp_size <= 0 or pp_size <= 0 or not moe_layer_pattern or len(moe_layer_pattern) % pp_size:
        raise ValueError("Qualification requires a nonempty uniform pipeline partition")
    if any(value not in (0, 1) for value in moe_layer_pattern):
        raise ValueError("Expected a binary dense/MoE layer pattern")
    receipts = validate_rank_receipts(receipts, tp_size * pp_size)
    expected_pairs = {(stage, rank) for stage in range(pp_size) for rank in range(tp_size)}
    if {(item["pipeline_rank"], item["tp_rank"]) for item in receipts} != expected_pairs:
        raise ValueError("Policy evidence must cover every pipeline and tensor-parallel rank")
    layers_per_stage = len(moe_layer_pattern) // pp_size
    for item in receipts:
        start = item["pipeline_rank"] * layers_per_stage
        layers = list(range(start + 1, start + layers_per_stage + 1))
        pattern = moe_layer_pattern[start : start + layers_per_stage]
        categories = {
            "attention": True,
            "dense_mlp": 0 in pattern,
            "routed_expert": 1 in pattern,
            "shared_expert": 1 in pattern,
        }
        if item["layer_numbers"] != layers or item["coverage"]["categories"] != categories or not item["passed"]:
            raise ValueError("Policy layer or adapter coverage differs from the expected pipeline stage")
    return receipts
