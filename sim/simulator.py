from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from scenario_io import DEFAULT_SCENARIO_DIR, resolve_scenario_dir
from topology import SimpleDiGraph
from topology_io import load_topology_json

GCL_XLSX = "GCL.xlsx"

SIM_HORIZON_S = 0.002

DEFAULT_LINK_BPS = 100e6


def parse_bitrate_mbps_or_default(val) -> float:
    try:
        return float(val) * 1e6
    except Exception:
        return DEFAULT_LINK_BPS


def parse_bytes(s: str) -> int:
    if s is None:
        return 1000
    t = str(s).strip().lower().replace("bytes", "b")
    if t.endswith("b"):
        t = t[:-1]
    try:
        return int(float(t))
    except Exception:
        return 1000


_TIME_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(ns|us|µs|ms|s)?\s*$", re.IGNORECASE)


def parse_time_seconds(s: str) -> float:
    if s is None:
        return 0.0
    m = _TIME_RE.match(str(s))
    if not m:
        try:
            return float(s)
        except Exception:
            return 0.0
    val = float(m.group(1))
    unit = (m.group(2) or "us").lower()
    if unit == "ns":
        return val * 1e-9
    if unit in ("us", "µs"):
        return val * 1e-6
    if unit == "ms":
        return val * 1e-3
    return val


def parse_durations_list(s: str) -> List[float]:
    if s is None:
        return []
    t = str(s).strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1]
    parts = [p.strip() for p in t.split(",") if p.strip()]
    return [parse_time_seconds(p) for p in parts]


@dataclass
class GateSchedule:
    cycle_time: float
    open_intervals: List[Tuple[float, float]]

    def next_open_time(self, t: float) -> float:
        if not self.open_intervals:
            return float("inf")
        modt = t % self.cycle_time
        for s, e in self.open_intervals:
            if modt < s:
                return t - modt + s
            if s <= modt < e:
                return t
        first_s = self.open_intervals[0][0]
        return t - modt + self.cycle_time + first_s

    def earliest_start_finish_allowed(self, t: float) -> float:
        return self.next_open_time(t)


_SHEET_PATTERNS = [
    re.compile(r"^(?P<node>.+?)[:\.\s_]+p(?P<port>\d+)$", re.IGNORECASE),
    re.compile(r"^(?P<node>.+?)[:\.\s_]+port[:\.\s_]+(?P<port>\d+)$", re.IGNORECASE),
]


def _extract_node_port_from_sheetname(name: str) -> Optional[Tuple[str, int]]:
    n = name.strip()
    for pat in _SHEET_PATTERNS:
        m = pat.match(n)
        if m:
            try:
                return m.group("node").strip(), int(m.group("port"))
            except Exception:
                pass
    return None


def load_gcls_from_excel(path: str) -> Dict[Tuple[str, int, int], GateSchedule]:
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot open GCL workbook '{path}': {e}")

    schedules: Dict[Tuple[str, int, int], GateSchedule] = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        node_port = _extract_node_port_from_sheetname(sheet)
        if node_port is None:
            raise ValueError(
                f"Cannot deduce node/port from sheet '{sheet}'. "
                "Use names like 'Central_Switch_p0'."
            )
        node, port = node_port

        cols = {str(c).strip().lower(): c for c in df.columns}
        offset_col = cols.get("offset") or cols.get("base-time") or cols.get("basetime")
        durations_col = cols.get("durations") or cols.get("segments") or cols.get("windows")
        qidx_col = cols.get("queueindex") or cols.get("queue") or cols.get("qidx") or cols.get("pcp")
        if not (offset_col and durations_col and qidx_col):
            raise ValueError(f"Sheet '{sheet}' must have columns: offset, durations, queueIndex")

        for q, sub in df.groupby(qidx_col):
            offsets = [parse_time_seconds(v) for v in sub[offset_col].dropna().astype(str).tolist()]
            durs_lists = [parse_durations_list(v) for v in sub[durations_col].dropna().astype(str).tolist()]
            if not durs_lists:
                continue
            durations: List[float] = []
            for dl in durs_lists:
                durations.extend(dl)
            if not durations:
                continue
            offset = offsets[0] if offsets else 0.0
            T = sum(durations)
            if T <= 0:
                continue

            open_intervals: List[Tuple[float, float]] = []
            t = offset % T
            is_open = True
            for d in durations:
                start = t % T
                end = (t + d) % T
                if is_open:
                    if start < end:
                        open_intervals.append((start, end))
                    else:
                        open_intervals.append((start, T))
                        open_intervals.append((0.0, end))
                t += d
                is_open = not is_open

            open_intervals.sort()
            merged: List[Tuple[float, float]] = []
            for s, e in open_intervals:
                if not merged:
                    merged.append((s, e))
                else:
                    ls, le = merged[-1]
                    if s <= le:
                        merged[-1] = (ls, max(le, e))
                    else:
                        merged.append((s, e))
            schedules[(node, int(port), int(q))] = GateSchedule(cycle_time=T, open_intervals=merged)
    return schedules


@dataclass
class Stream:
    name: str
    src: str
    dst: str
    interval: float
    size_bytes: int
    deadline: float
    qidx: int


def load_streams_csv(path: str) -> List[Stream]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot open streams CSV '{path}': {e}")

    def col(*names):
        for n in names:
            if n in df.columns:
                return n
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for n in names:
            c = lower_map.get(n.lower())
            if c:
                return c
        return None

    c_name = col("name", "id")
    c_src = col("sourceId", "source", "src", "talker")
    c_dst = col("destId", "destination", "dst", "listener")
    c_int = col("interval", "period", "T")
    c_size = col("packetSize", "size", "frame", "frame_size")
    c_dead = col("deadline", "d")
    c_q = col("trafficClass", "queueIndex", "pcp", "qidx", "queue")
    if not all([c_name, c_src, c_dst, c_int, c_size]):
        raise ValueError("CSV must include name/id, talker, listener, period, frame_size")

    streams: List[Stream] = []
    for _, row in df.iterrows():
        name = str(row[c_name])
        src = str(row[c_src])
        dst = str(row[c_dst])
        interval = parse_time_seconds(row[c_int])
        size_b = parse_bytes(row[c_size])
        deadline = parse_time_seconds(row[c_dead]) if c_dead else 0.0
        qidx = int(row[c_q]) if c_q and pd.notna(row[c_q]) else 0
        if interval <= 0:
            continue
        streams.append(Stream(name, src, dst, interval, size_b, deadline, qidx))
    return streams


def allocated_time_over_horizon(
    open_intervals: List[Tuple[float, float]], T: float, horizon: float
) -> float:
    if T <= 0 or horizon <= 0 or not open_intervals:
        return 0.0
    per_cycle = sum(e - s for s, e in open_intervals)
    full = int(horizon // T)
    rem = horizon - full * T
    alloc = full * per_cycle
    for s, e in open_intervals:
        if rem <= 0:
            break
        alloc += max(0.0, min(rem, e) - s)
    return alloc


def overlap_with_open(schedule: GateSchedule, start: float, end: float) -> float:
    if start >= end or not schedule.open_intervals:
        return 0.0
    T = schedule.cycle_time
    total = 0.0
    t = start
    while t < end:
        modt = t % T
        if modt < 1e-12 or abs(modt - T) < 1e-12:
            modt = 0.0
        span = min(end - t, T - modt)
        if span < 1e-12:
            span = 1e-12
        win_end = modt + span
        for s, e in schedule.open_intervals:
            a = max(modt, s)
            b = min(win_end, e)
            if b > a:
                total += b - a
        t += span
    return total


def _build_port_unions(
    gcls: Dict[Tuple[str, int, int], GateSchedule]
) -> Dict[Tuple[str, int], Tuple[float, List[Tuple[float, float]]]]:
    per_port_queues: Dict[Tuple[str, int], List[GateSchedule]] = defaultdict(list)
    for (node, port, _q), sched in gcls.items():
        per_port_queues[(node, port)].append(sched)

    port_union: Dict[Tuple[str, int], Tuple[float, List[Tuple[float, float]]]] = {}
    for (node, port), scheds in per_port_queues.items():
        Ts = {round(s.cycle_time, 12) for s in scheds}
        T = max(s.cycle_time for s in scheds) if len(Ts) > 1 else scheds[0].cycle_time
        pts: List[float] = []
        for s in scheds:
            pts.extend([a % T for a, b in s.open_intervals])
            pts.extend([b % T for a, b in s.open_intervals])
        pts = sorted(set(pts))
        if not pts:
            port_union[(node, port)] = (T, [])
            continue
        atoms: List[Tuple[float, float]] = []
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            if b <= a:
                if any(any(x <= a < y for x, y in s.open_intervals) for s in scheds):
                    atoms.append((a, T))
                if b > 0 and any(any(x <= 0 < y for x, y in s.open_intervals) for s in scheds):
                    atoms.append((0.0, b))
            else:
                if any(any(x <= (a + 1e-15) < y for x, y in s.open_intervals) for s in scheds):
                    atoms.append((a, b))
        atoms.sort()
        merged: List[Tuple[float, float]] = []
        for a, b in atoms:
            if not merged:
                merged.append((a, b))
            else:
                la, lb = merged[-1]
                if a <= lb:
                    merged[-1] = (la, max(lb, b))
                else:
                    merged.append((a, b))
        port_union[(node, port)] = (T, merged)
    return port_union


def run_simulation(
    scenario_dir: str = str(DEFAULT_SCENARIO_DIR),
    gcl_path: str = GCL_XLSX,
) -> Dict[str, float]:
    scenario = resolve_scenario_dir(scenario_dir)
    G: SimpleDiGraph = load_topology_json(scenario.topology)
    gcls = load_gcls_from_excel(gcl_path)
    streams = load_streams_csv(str(scenario.csv))
    if not streams:
        return {"packets_total": 0, "success_rate": 0.0, "avg_delay_ms": 0.0, "ou_percent": 0.0}

    sim_end = SIM_HORIZON_S
    eps = 1e-15

    next_free_queue: Dict[Tuple[str, int, int], float] = defaultdict(float)
    next_free_port: Dict[Tuple[str, int], float] = defaultdict(float)
    per_packet: List[Tuple[str, int, float, float, float, bool]] = []
    occupied_overlap: Dict[Tuple[str, int], float] = defaultdict(float)
    port_union = _build_port_unions(gcls)

    for s in streams:
        path = G.shortest_path(s.src, s.dst)
        if not path or len(path) < 2:
            continue
        t = 0.0
        pkt_idx = 0
        while t < sim_end - eps:
            pkt_idx += 1
            gen_time = t
            cur_time = gen_time
            delivered = True
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = G[u].get(v)
                if edge is None:
                    delivered = False
                    break
                link_bps = parse_bitrate_mbps_or_default(edge.get("capacity_mbps", 100.0))
                egress_port = int(edge.get("portA", 0))
                q = s.qidx
                tx_time = (s.size_bytes * 8.0) / link_bps
                key_q = (u, egress_port, q)
                key_port = (u, egress_port)
                base = max(cur_time, next_free_queue[key_q], next_free_port[key_port])
                sched = gcls.get((u, egress_port, q))
                if sched is not None and not sched.open_intervals:
                    delivered = False
                    break
                start_tx = sched.earliest_start_finish_allowed(base) if sched is not None else base
                finish_tx = start_tx + tx_time
                next_free_queue[key_q] = finish_tx
                next_free_port[key_port] = finish_tx
                if sched is not None:
                    occupied_overlap[(u, egress_port)] += overlap_with_open(sched, start_tx, finish_tx)
                else:
                    occupied_overlap[(u, egress_port)] += finish_tx - start_tx
                cur_time = finish_tx
                if cur_time > sim_end:
                    delivered = False
                    break
            if delivered:
                delay = cur_time - gen_time
                on_time = (s.deadline <= 0) or (delay <= s.deadline)
                per_packet.append((s.name, pkt_idx, gen_time, cur_time, delay, on_time))
            t += s.interval

    if per_packet:
        ontime = sum(1 for *_, ok in per_packet if ok)
        total = len(per_packet)
        avg_ms = (sum(d for *_, d, _ in per_packet) / total) * 1e3
        sr = ontime / total
    else:
        total = 0
        sr = 0.0
        avg_ms = 0.0

    allocated_total = 0.0
    occupied_total = 0.0
    for (node, port), (T, intervals) in port_union.items():
        allocated_total += allocated_time_over_horizon(intervals, T, sim_end)
        occupied_total += occupied_overlap.get((node, port), 0.0)
    ou = (occupied_total / allocated_total * 100.0) if allocated_total > 0 else 0.0

    return {
        "packets_total": total,
        "success_rate": sr,
        "avg_delay_ms": avg_ms,
        "ou_percent": ou,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the TAS network performance model.")
    parser.add_argument(
        "--scenario",
        default=str(DEFAULT_SCENARIO_DIR),
        help="Scenario directory containing a streams CSV and topology.json.",
    )
    parser.add_argument(
        "--gcl",
        default=GCL_XLSX,
        help=f"GCL workbook to evaluate (default: {GCL_XLSX}).",
    )
    args = parser.parse_args(argv)

    metrics = run_simulation(args.scenario, args.gcl)
    print("=== SIMULATION SUMMARY ===")
    print(f"Packets delivered      : {metrics['packets_total']}")
    print(f"Success rate           : {metrics['success_rate'] * 100:.2f}%")
    print(f"Average delay          : {metrics['avg_delay_ms']:.3f} ms")
    print(f"Overall utilisation    : {metrics['ou_percent']:.2f}%")


if __name__ == "__main__":
    sys.exit(main())
