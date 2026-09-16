"""Validate physical receiver evidence across independent inference engines."""


def validate_receivers(receipts, engine_ids, tp_size):
    if not engine_ids or len(set(engine_ids)) != len(engine_ids) or any(not engine for engine in engine_ids):
        raise ValueError("Expected inference engine identities must be nonempty and unique")
    if tp_size <= 0:
        raise ValueError("Expected tensor-parallel size must be positive")
    expected = {(engine, rank) for engine in engine_ids for rank in range(tp_size)}
    observed = {(item["engine_id"], item["tp_rank"]) for item in receipts}
    if not expected or len(receipts) != len(expected) or observed != expected:
        raise ValueError("Receiver audit must cover every engine and tensor-parallel rank exactly once")
    for engine in engine_ids:
        identities = {item["adapter_id"] for item in receipts if item["engine_id"] == engine}
        if len(identities) != 1 or next(iter(identities)) <= 0:
            raise ValueError("Receiver ranks disagree on the active adapter")
    for item in receipts:
        if item["context"] != 32768 or item["model_dtype"] != "torch.bfloat16":
            raise ValueError("Receiver model/context contract differs")
        for tensors in (list(item["buffers"].values()), item["kv_tensors"]):
            if not tensors or any(tensor["dtype"] != "torch.bfloat16" for tensor in tensors):
                raise ValueError("Receiver buffers and actual KV tensors must be BF16")
    return sorted(receipts, key=lambda item: (item["engine_id"], item["tp_rank"]))


def fingerprint_receivers(receipts, engine_ids, tp_size):
    checked = validate_receivers(receipts, engine_ids, tp_size)
    fingerprints = {engine: {} for engine in sorted(engine_ids)}
    for item in checked:
        fingerprints[item["engine_id"]][item["tp_rank"]] = {
            name: descriptor["sha256"] for name, descriptor in sorted(item["buffers"].items())
        }
    return fingerprints
