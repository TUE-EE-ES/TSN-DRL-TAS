from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from topology import SimpleDiGraph
from topology_io import load_topology_json, topology_signature

DEG_NORM_MAX = 7.0
INDEG_NORM_MAX = 7.0

DEFAULT_W_SR = 3.0
DEFAULT_W_OU = 3.0
DEFAULT_W_DELAY = 0.15

_TIME_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(ns|us|µs|ms|s)?\s*$", re.IGNORECASE)


def reward_from_metrics(
    sr: float,
    ou: float,
    avg_ms: float,
    w_sr: float = DEFAULT_W_SR,
    w_ou: float = DEFAULT_W_OU,
    w_delay: float = DEFAULT_W_DELAY,
) -> float:
    return w_sr * sr + w_ou * (ou / 100.0) - w_delay * avg_ms


def parse_time_seconds_with_us_default(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0.0
    s = str(val).strip().replace("Âµs", "us")
    m = _TIME_RE.match(s)
    if not m:
        try:
            return float(s)
        except Exception:
            return 0.0
    num = float(m.group(1))
    unit = (m.group(2) or "us").lower()
    if unit == "ns":
        return num * 1e-9
    if unit in ("us", "µs"):
        return num * 1e-6
    if unit == "ms":
        return num * 1e-3
    if unit == "s":
        return num
    return num * 1e-6


def to_us_int(seconds: float) -> int:
    return max(1, int(round(seconds * 1e6)))


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else max(a, b)


def lcm_list(vals: List[int]) -> int:
    v = vals[0]
    for x in vals[1:]:
        v = lcm(v, x)
    return v


def sanitize_sheet_name(name: str) -> str:
    bad = r"[]:*?/\\"
    return name.translate({ord(c): "_" for c in bad})[:31]


def compute_node_depths(
    graph: SimpleDiGraph, node_names: List[str], incoming: Dict[str, List[str]]
) -> Dict[str, int]:
    root = next(
        (
            n
            for n in node_names
            if str(graph.node_attributes(n).get("role", "")).lower() == "central"
        ),
        node_names[0] if node_names else None,
    )
    depths: Dict[str, int] = {n: -1 for n in node_names}
    if root is None:
        return depths
    depths[root] = 0
    q: deque[str] = deque([root])
    while q:
        u = q.popleft()
        neighbours = set(graph[u].keys()) | set(incoming.get(u, []))
        for v in neighbours:
            if depths[v] == -1:
                depths[v] = depths[u] + 1
                q.append(v)
    max_depth = max((d for d in depths.values() if d >= 0), default=0)
    for n, d in depths.items():
        if d < 0:
            depths[n] = max_depth + 1
    return depths


@dataclass
class PortInfo:
    node: str
    port: int
    hp_us: int
    queues: List[int]


@dataclass
class QueueSpec:
    port_idx: int
    queue: int
    hp_us: int
    load_norm: float


@dataclass
class RolloutSample:
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    node_feat: torch.Tensor
    queue_feat: torch.Tensor
    stats: torch.Tensor
    queue_idx: torch.Tensor
    action_raw: torch.Tensor
    tau: torch.Tensor
    B: torch.Tensor
    logp_old: torch.Tensor
    value_old: torch.Tensor
    entropy: torch.Tensor
    reward: Optional[torch.Tensor] = None


class TasScenario:
    def __init__(self, scenario_dir: str):
        from scenario_io import resolve_scenario_dir

        scenario = resolve_scenario_dir(scenario_dir)
        self.scenario_dir = str(scenario.dir)
        self.stream_path = str(scenario.csv)
        self.topology_path = str(scenario.topology)
        self.name = scenario.name
        self.G: SimpleDiGraph = load_topology_json(self.topology_path)
        self.topology_sig = topology_signature(self.G)
        self.df_streams = pd.read_csv(self.stream_path)

        self.streams: List[Tuple[str, str, str, float, int]] = []
        for idx, row in self.df_streams.iterrows():
            src = str(row.get("talker"))
            dst = str(row.get("listener"))
            T = parse_time_seconds_with_us_default(row.get("period"))
            if T <= 0:
                continue
            q = int(row.get("queue", 0)) if not pd.isna(row.get("queue", 0)) else 0
            name = str(row.get("id", f"Flow_{idx}"))
            self.streams.append((name, src, dst, T, q))

        port_periods: Dict[Tuple[str, int], List[int]] = {}
        queue_usage: Dict[Tuple[str, int, int], int] = {}
        port_load_counts: Dict[Tuple[str, int], int] = {}
        self.paths: Dict[Tuple[str, str], Optional[List[str]]] = {}
        for name, src, dst, T, q in self.streams:
            path = self.paths.get((src, dst))
            if path is None:
                path = self.G.shortest_path(src, dst)
                self.paths[(src, dst)] = path
            if not path or len(path) < 2:
                continue
            T_us = to_us_int(T)
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if str(self.G.node_attributes(u).get("type", "")).lower() != "switch":
                    continue
                edge = self.G[u].get(v)
                if edge is None:
                    continue
                egress_port = int(edge.get("portA", 0))
                port_periods.setdefault((u, egress_port), []).append(T_us)
                queue_usage[(u, egress_port, q)] = queue_usage.get((u, egress_port, q), 0) + 1
                port_load_counts[(u, egress_port)] = port_load_counts.get((u, egress_port), 0) + 1

        port_queues: Dict[Tuple[str, int], List[int]] = {}
        for (node, port) in port_periods.keys():
            qs = set()
            for _, src, dst, _, q in self.streams:
                path = self.paths.get((src, dst)) or []
                for i in range(len(path) - 1):
                    if path[i] == node:
                        edge = self.G[path[i]].get(path[i + 1])
                        if edge and int(edge.get("portA", 0)) == port:
                            qs.add(q)
            port_queues[(node, port)] = sorted(qs) if qs else [0]

        self.port_list: List[PortInfo] = []
        hp_vals: List[int] = []
        for (node, port), plist in sorted(port_periods.items()):
            hp = lcm_list(plist) if plist else 5000
            hp_vals.append(hp)
            self.port_list.append(
                PortInfo(node=node, port=port, hp_us=hp, queues=port_queues[(node, port)])
            )
        self.hp_max = max(hp_vals) if hp_vals else 1

        self.node_names = list(self.G.nodes())
        self.node_index = {n: i for i, n in enumerate(self.node_names)}
        N = len(self.node_names)
        deg = np.zeros((N, 1), dtype=np.float32)
        indeg = np.zeros((N, 1), dtype=np.float32)
        is_switch = np.zeros((N, 1), dtype=np.float32)
        is_controller = np.zeros((N, 1), dtype=np.float32)
        is_sensor = np.zeros((N, 1), dtype=np.float32)
        est_load = np.zeros((N, 1), dtype=np.float32)

        incoming_map: Dict[str, List[str]] = {n: [] for n in self.node_names}
        for u in self.node_names:
            for v in self.G[u].keys():
                incoming_map[v].append(u)

        for i, n in enumerate(self.node_names):
            outs = list(self.G[n].keys())
            deg[i, 0] = len(outs)
            indeg[i, 0] = len(incoming_map[n])
            node_type = str(self.G.node_attributes(n).get("type", "")).lower()
            role = str(self.G.node_attributes(n).get("role", "")).lower()
            is_switch[i, 0] = 1.0 if node_type == "switch" else 0.0
            is_controller[i, 0] = 1.0 if role == "controller" else 0.0
            is_sensor[i, 0] = 1.0 if role == "sensor" else 0.0

        for _, src, dst, _, _ in self.streams:
            path = self.paths.get((src, dst)) or []
            for node in path[:-1]:
                est_load[self.node_index[node], 0] += 1.0

        deg_max = max(1.0, float(DEG_NORM_MAX))
        indeg_max = max(1.0, float(INDEG_NORM_MAX))
        load_max = max(1.0, est_load.max())
        self.X_base = torch.tensor(
            np.hstack(
                [
                    is_switch,
                    is_controller,
                    is_sensor,
                    deg / deg_max,
                    indeg / indeg_max,
                    est_load / load_max,
                ]
            ),
            dtype=torch.float32,
        )

        A = np.zeros((N, N), dtype=np.float32)
        for i, u in enumerate(self.node_names):
            for v in self.G[u].keys():
                A[i, self.node_index[v]] = 1.0
        A = A + np.eye(N, dtype=np.float32)
        D_inv = np.diag(1.0 / np.maximum(1.0, A.sum(axis=1)))
        self.A_norm = torch.tensor(D_inv @ A, dtype=torch.float32)
        self.node_depths = compute_node_depths(self.G, self.node_names, incoming_map)
        depth_max = max((d for d in self.node_depths.values()), default=1)
        depth_norm = {
            n: (self.node_depths[n] / depth_max) if depth_max else 0.0 for n in self.node_names
        }
        port_load_max = max(1.0, float(max(port_load_counts.values(), default=0)))

        edge_index_pairs: List[List[int]] = []
        edge_attr_list: List[List[float]] = []
        for u in self.node_names:
            outs = self.G[u]
            out_degree = max(1.0, float(len(outs)))
            src_idx = self.node_index[u]
            src_type = str(self.G.node_attributes(u).get("type", "")).lower()
            src_switch = 1.0 if src_type == "switch" else 0.0
            for v, attr in outs.items():
                dst_idx = self.node_index[v]
                dst_type = str(self.G.node_attributes(v).get("type", "")).lower()
                dst_switch = 1.0 if dst_type == "switch" else 0.0
                cap = float(attr.get("capacity_mbps", 100.0))
                cap_norm = cap / 100.0
                port = int(attr.get("portA", 0))
                load = port_load_counts.get((u, port), 0)
                load_norm = load / port_load_max
                edge_index_pairs.append([src_idx, dst_idx])
                edge_attr_list.append(
                    [
                        cap_norm,
                        load_norm,
                        depth_norm.get(u, 0.0),
                        depth_norm.get(v, 0.0),
                        1.0 / out_degree,
                        src_switch,
                        dst_switch,
                    ]
                )

        feat_dim = len(edge_attr_list[0]) if edge_attr_list else 7
        if edge_index_pairs:
            edge_index = torch.tensor(edge_index_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_attr = torch.zeros((1, feat_dim), dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_dim = feat_dim

        self.queue_meta: List[QueueSpec] = []
        self.queue_node_idx: List[int] = []
        queue_features: List[List[float]] = []
        queue_loads = list(queue_usage.values()) or [1.0]
        q_load_max = max(queue_loads)

        for p_idx, port in enumerate(self.port_list):
            node_idx = self.node_index[port.node]
            for q in port.queues:
                load = queue_usage.get((port.node, port.port, q), 1)
                load_norm = load / q_load_max
                self.queue_meta.append(
                    QueueSpec(port_idx=p_idx, queue=q, hp_us=port.hp_us, load_norm=load_norm)
                )
                self.queue_node_idx.append(node_idx)
                queue_features.append(
                    [
                        q / 7.0,
                        load_norm,
                        port.hp_us / self.hp_max,
                        len(port.queues) / 8.0,
                        1.0 if q == 7 else 0.0,
                    ]
                )

        if not self.queue_meta:
            self.queue_meta.append(QueueSpec(port_idx=0, queue=0, hp_us=5000, load_norm=1.0))
            self.queue_node_idx.append(0)
            queue_features.append([0.0, 1.0, 1.0, 0.0, 0.0])

        self.queue_features = torch.tensor(queue_features, dtype=torch.float32)
        self.queue_node_idx = torch.tensor(self.queue_node_idx, dtype=torch.long)

        periods = [to_us_int(T) for _, _, _, T, _ in self.streams] or [2000]
        deadlines = [
            float(row.get("deadline", 1000))
            for _, row in self.df_streams.iterrows()
            if pd.notna(row.get("deadline"))
        ] or [1000.0]

        self.scenario_stats = torch.tensor(
            [
                len(self.streams) / 1000.0,
                len(self.port_list) / 50.0,
                len(self.queue_meta) / 400.0,
                np.mean(periods) / 2000.0,
                min(deadlines) / 1000.0,
            ],
            dtype=torch.float32,
        )

    def tensors(
        self, noise_std: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X = self.X_base.clone()
        queue_feat = self.queue_features.clone()
        stats = self.scenario_stats.clone()
        if noise_std > 0:
            X += torch.randn_like(X) * noise_std
            queue_feat += torch.randn_like(queue_feat) * noise_std
            stats += torch.randn_like(stats) * (noise_std * 0.5)
        return self.A_norm, X, queue_feat, stats


def write_gcl_from_actions(
    port_list: List[PortInfo],
    queue_meta: List[QueueSpec],
    actions: List[Tuple[int, int]],
    n_slots: int,
    out_xlsx: str,
) -> None:
    per_port: Dict[int, List[Tuple[int, int, int]]] = {idx: [] for idx in range(len(port_list))}
    for (offset_idx, duty_slots), spec in zip(actions, queue_meta):
        per_port[spec.port_idx].append((spec.queue, offset_idx, duty_slots))

    writer = pd.ExcelWriter(out_xlsx, engine="xlsxwriter")
    for port_idx, info in enumerate(port_list):
        slot_us = max(1, info.hp_us // n_slots)
        rows = []
        action_map = {queue: (offset, duty) for queue, offset, duty in per_port.get(port_idx, [])}
        for q in range(8):
            if q in action_map:
                offset_idx, duty_slots = action_map[q]
                duty_slots = max(1, min(n_slots, duty_slots))
                offset_idx = offset_idx % n_slots
                open_len = duty_slots * slot_us
                offset_us = offset_idx * slot_us
                durations = [open_len, max(1, info.hp_us - open_len)]
            else:
                offset_us = 0
                durations = [0, info.hp_us]
            rows.append(
                {
                    "offset": f"{offset_us}us",
                    "durations": "[" + ", ".join(f"{d}us" for d in durations) + "]",
                    "queueIndex": q,
                }
            )
        pd.DataFrame(rows, columns=["offset", "durations", "queueIndex"]).to_excel(
            writer, index=False, sheet_name=sanitize_sheet_name(f"{info.node}_p{info.port}")
        )
    writer.close()


def _round_to_sum(values: np.ndarray, target: int) -> List[int]:
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = int(target - floored.sum())
    if deficit > 0:
        indices = np.argsort(-remainders)
        for i in range(min(deficit, len(floored))):
            floored[indices[i]] += 1
    elif deficit < 0:
        indices = np.argsort(remainders)
        for i in range(min(-deficit, len(floored))):
            floored[indices[i]] -= 1
    return floored.tolist()


def compile_gcl_from_template(
    tau: torch.Tensor,
    B: torch.Tensor,
    port_list: List[PortInfo],
    queue_meta: List[QueueSpec],
    out_xlsx: str,
    min_open_us: int = 1,
) -> None:
    tau_np = tau.detach().cpu().numpy().astype(np.float64)
    B_np = B.detach().cpu().numpy().astype(np.float64)
    K = len(tau_np)
    P = B_np.shape[1]

    writer = pd.ExcelWriter(out_xlsx, engine="xlsxwriter")
    for info in port_list:
        T = info.hp_us
        phase_lens = _round_to_sum(tau_np * T, T)
        active_queues = set(info.queues)
        rows = []
        for c in range(8):
            durations: List[int] = []
            if c in active_queues:
                for k in range(K):
                    pl = phase_lens[k]
                    if pl <= 0:
                        continue
                    open_us = int(round(B_np[k, min(c, P - 1)] * pl))
                    open_us = max(0, min(pl, open_us))
                    if open_us < min_open_us:
                        open_us = 0
                    durations.append(open_us)
                    durations.append(pl - open_us)
                while len(durations) > 1 and durations[-1] == 0:
                    durations.pop()

            if not durations:
                durations = [0, max(1, T)]

            total = sum(durations)
            if total < T:
                durations.append(T - total)
            elif total > T:
                durations[-1] = max(0, durations[-1] - (total - T))

            rows.append(
                {
                    "offset": "0us",
                    "durations": "[" + ", ".join(f"{d}us" for d in durations) + "]",
                    "queueIndex": c,
                }
            )
        pd.DataFrame(rows, columns=["offset", "durations", "queueIndex"]).to_excel(
            writer, index=False, sheet_name=sanitize_sheet_name(f"{info.node}_p{info.port}")
        )
    writer.close()
