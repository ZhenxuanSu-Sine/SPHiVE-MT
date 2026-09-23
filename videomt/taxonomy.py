import json

import torch


class Taxonomy:
    """Minimal tree taxonomy used by SPHiVE-MT.

    V1 requires integer node ids to be contiguous in [0, num_nodes).
    """

    def __init__(self, nodes):
        self.nodes = sorted(nodes, key=lambda x: int(x["id"]))
        ids = [int(node["id"]) for node in self.nodes]
        if ids != list(range(len(ids))):
            raise ValueError(
                "SPHiVE taxonomy v1 requires contiguous node ids: 0..N-1"
            )

        self.num_nodes = len(self.nodes)
        self.parent = [-1] * self.num_nodes
        for node in self.nodes:
            node_id = int(node["id"])
            parent = node.get("parent")
            self.parent[node_id] = -1 if parent is None else int(parent)

        descendants = torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=torch.bool
        )
        for node_id in range(self.num_nodes):
            current = node_id
            while current >= 0:
                descendants[current, node_id] = True
                current = self.parent[current]
        self.descendant_matrix = descendants

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data["nodes"] if isinstance(data, dict) else data
        return cls(nodes)

    def descendants(self, node_id):
        return torch.nonzero(
            self.descendant_matrix[int(node_id)], as_tuple=False
        ).flatten()

    def to(self, device):
        self.descendant_matrix = self.descendant_matrix.to(device)
        return self
