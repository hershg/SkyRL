"""Bind diagnostic GPU bundles to explicitly admitted role nodes."""


def validate_role_nodes(nodes, role_addresses, gpus_per_node):
    if set(role_addresses) != {"trainer", "inference"}:
        raise ValueError("Qualification requires trainer and inference roles")
    addresses = [address for role in role_addresses.values() for address in role]
    if len(role_addresses["trainer"]) != 3 or len(role_addresses["inference"]) != 2 or len(set(addresses)) != 5:
        raise ValueError("Qualification requires three trainer and two inference nodes")
    alive = [node for node in nodes if node["Alive"]]
    if len(alive) != 5 or {node["NodeManagerAddress"] for node in alive} != set(addresses):
        raise ValueError("Ray membership differs from the five admitted nodes")
    by_address = {node["NodeManagerAddress"]: node for node in alive}
    for address in addresses:
        resources = by_address[address]["Resources"]
        if resources.get("GPU") != gpus_per_node or f"node:{address}" not in resources:
            raise ValueError("Ray resources differ from the admitted node")
    return {role: [by_address[address]["NodeID"] for address in values] for role, values in role_addresses.items()}


def create_role_placement_groups(role_addresses, gpus_per_node=8):
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group

    from skyrl.train.utils.utils import ResolvedPlacementGroup

    expected = validate_role_nodes(ray.nodes(), role_addresses, gpus_per_node)
    groups = {}
    try:
        for role, addresses in role_addresses.items():
            groups[role] = placement_group(
                [{"GPU": 1, "CPU": 1, f"node:{address}": 0.001} for address in addresses for _ in range(gpus_per_node)],
                strategy="PACK",
            )
            ray.get(groups[role].ready(), timeout=120)
        resolved = {role: ResolvedPlacementGroup(group) for role, group in groups.items()}
        for role, group in resolved.items():
            if group.bundle_node_ids != [node for node in sorted(expected[role]) for _ in range(gpus_per_node)]:
                raise ValueError("Resolved GPU bundles differ from admitted role placement")
        return resolved
    except BaseException:
        for group in groups.values():
            remove_placement_group(group)
        raise
