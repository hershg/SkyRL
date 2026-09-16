import copy

import pytest

from examples.model_checks.placement import validate_role_nodes


def make_nodes():
    return [
        {
            "Alive": True,
            "NodeID": f"id-{index}",
            "NodeManagerAddress": f"host-{index}",
            "Resources": {"GPU": 8, f"node:host-{index}": 1},
        }
        for index in range(5)
    ]


def test_admitted_role_order_preserved():
    roles = {"trainer": ["host-2", "host-0", "host-1"], "inference": ["host-4", "host-3"]}
    assert validate_role_nodes(make_nodes(), roles, 8) == {
        "trainer": ["id-2", "id-0", "id-1"],
        "inference": ["id-4", "id-3"],
    }


@pytest.mark.parametrize("failure", ["missing", "extra", "dead", "gpu", "resource", "duplicate"])
def test_rejects_unadmitted_membership(failure):
    nodes = make_nodes()
    roles = {"trainer": [f"host-{i}" for i in range(3)], "inference": [f"host-{i}" for i in range(3, 5)]}
    if failure == "missing":
        nodes.pop()
    elif failure == "extra":
        nodes.append(copy.deepcopy(nodes[0]))
    elif failure == "dead":
        nodes[0]["Alive"] = False
    elif failure == "gpu":
        nodes[0]["Resources"]["GPU"] = 7
    elif failure == "resource":
        del nodes[0]["Resources"]["node:host-0"]
    else:
        roles["inference"][0] = roles["trainer"][0]
    with pytest.raises(ValueError):
        validate_role_nodes(nodes, roles, 8)


def test_rejects_wrong_role_split_even_with_five_distinct_nodes():
    roles = {"trainer": ["host-0", "host-1"], "inference": ["host-2", "host-3", "host-4"]}
    with pytest.raises(ValueError, match="three trainer and two inference"):
        validate_role_nodes(make_nodes(), roles, 8)
