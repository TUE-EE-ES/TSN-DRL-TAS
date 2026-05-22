from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from topology import SimpleDiGraph


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def graph_to_dict(G: SimpleDiGraph) -> Dict[str, Any]:
    nodes = []
    for node in sorted(G.nodes()):
        attrs = _to_builtin(dict(G.node_attributes(node)))
        nodes.append({"id": node, "attrs": attrs})

    edges = []
    for u in sorted(G.keys()):
        for v in sorted(G[u].keys()):
            attrs = _to_builtin(dict(G[u][v]))
            edges.append({"u": u, "v": v, "attrs": attrs})

    return {"nodes": nodes, "edges": edges}


def write_topology_json(G: SimpleDiGraph, path: str | Path) -> None:
    payload = graph_to_dict(G)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_topology_json(path: str | Path) -> SimpleDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    G = SimpleDiGraph()
    for node in nodes:
        node_id = str(node.get("id"))
        attrs = node.get("attrs", {}) or {}
        G.add_node(node_id, **attrs)

    for edge in edges:
        u = str(edge.get("u"))
        v = str(edge.get("v"))
        attrs = edge.get("attrs", {}) or {}
        G.add_edge(u, v, **attrs)

    return G


def topology_signature(G: SimpleDiGraph) -> str:
    payload = graph_to_dict(G)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
