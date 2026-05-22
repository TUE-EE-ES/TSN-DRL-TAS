from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from drl.environment import (
    TasScenario,
    parse_time_seconds_with_us_default,
    reward_from_metrics,
    to_us_int,
    write_gcl_from_actions,
)
from scenario_io import DEFAULT_SCENARIO_PATTERN, discover_scenarios
from sim.simulator import GCL_XLSX, run_simulation


@dataclass
class FlowRecord:
    period_us: int
    deadline_us: int
    frame_bytes: float
    n_per_period: int
    capacity_mbps: float


class EDFScheduler:
    def __init__(
        self,
        scenario: TasScenario,
        n_slots: int = 8,
        guard: float = 0.10,
        util_scale: float = 0.5,
        slot_cap: float = 0.6,
    ) -> None:
        self.scenario = scenario
        self.n_slots = max(1, n_slots)
        self.guard = max(0.0, guard)
        self.util_scale = max(0.0, min(util_scale, 1.0))
        self.slot_cap = max(0.1, min(slot_cap, 1.0))
        self.df = scenario.df_streams
        self.columns = {str(c).strip().lower(): c for c in self.df.columns}
        self.port_index: Dict[Tuple[str, int], int] = {
            (info.node, info.port): idx for idx, info in enumerate(self.scenario.port_list)
        }
        self.port_queue_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, spec in enumerate(self.scenario.queue_meta):
            self.port_queue_indices[spec.port_idx].append(idx)
        self.port_queue_flows = self._collect_port_flows()

    def _col(self, *names: str) -> Optional[str]:
        for name in names:
            col = self.columns.get(name.lower())
            if col is not None:
                return col
        return None

    def _collect_port_flows(self) -> Dict[Tuple[int, int], List[FlowRecord]]:
        flows: Dict[Tuple[int, int], List[FlowRecord]] = defaultdict(list)
        col_src = self._col("talker", "sourceid", "src", "source")
        col_dst = self._col("listener", "destid", "dst", "destination")
        col_period = self._col("period", "interval")
        if not (col_src and col_dst and col_period):
            return flows

        col_queue = self._col("queue", "trafficclass", "queueindex", "pcp", "qidx")
        col_size = self._col("frame_size", "framesize", "packet_size", "packetsize", "size_bytes")
        col_deadline = self._col("deadline")
        col_n_per = self._col("n_per_period", "frames_per_period", "packets_per_period")

        for _, row in self.df.iterrows():
            src, dst = str(row[col_src]), str(row[col_dst])
            period_us = max(1, to_us_int(parse_time_seconds_with_us_default(row[col_period])))

            deadline_val = row[col_deadline] if col_deadline and pd.notna(row[col_deadline]) else None
            deadline_us = (
                to_us_int(parse_time_seconds_with_us_default(deadline_val))
                if deadline_val is not None
                else period_us
            )
            if deadline_us <= 0:
                deadline_us = period_us

            queue = int(row[col_queue]) if col_queue and pd.notna(row[col_queue]) else 0
            size_val = row[col_size] if col_size and pd.notna(row[col_size]) else 800
            try:
                frame_bytes = float(size_val)
            except Exception:
                frame_bytes = 800.0

            n_per_period = 1
            if col_n_per and pd.notna(row[col_n_per]):
                try:
                    n_per_period = max(1, int(round(float(row[col_n_per]))))
                except Exception:
                    n_per_period = 1

            path = self.scenario.paths.get((src, dst)) or self.scenario.G.shortest_path(src, dst)
            if not path or len(path) < 2:
                continue
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if str(self.scenario.G.node_attributes(u).get("type", "")).lower() != "switch":
                    continue
                edge = self.scenario.G[u].get(v)
                if edge is None:
                    continue
                port = int(edge.get("portA", 0))
                port_idx = self.port_index.get((u, port))
                if port_idx is None:
                    continue
                flows[(port_idx, queue)].append(
                    FlowRecord(
                        period_us=period_us,
                        deadline_us=deadline_us,
                        frame_bytes=frame_bytes,
                        n_per_period=n_per_period,
                        capacity_mbps=float(edge.get("capacity_mbps", 100.0)),
                    )
                )
        return flows

    def _required_open_time_us(self, hp_us: int, flows: Sequence[FlowRecord]) -> float:
        if hp_us <= 0 or not flows or self.util_scale <= 0.0:
            return 0.0
        total = 0.0
        for flow in flows:
            periods_in_hp = max(1, math.ceil(hp_us / flow.period_us))
            packets = periods_in_hp * flow.n_per_period
            cap_bps = max(1e3, flow.capacity_mbps * 1e6)
            serialization_us = (flow.frame_bytes * 8.0) / cap_bps * 1e6
            total += packets * serialization_us
        return total * self.util_scale * (1.0 + self.guard)

    def _schedule_port(self, port_idx: int, spec_indices: Iterable[int]) -> Dict[int, Tuple[int, int]]:
        info = self.scenario.port_list[port_idx]
        slot_us = max(1, info.hp_us // self.n_slots)
        queue_entries: List[Tuple[int, int, float, float]] = []
        for spec_idx in spec_indices:
            spec = self.scenario.queue_meta[spec_idx]
            flows = self.port_queue_flows.get((port_idx, spec.queue), [])
            required_us = self._required_open_time_us(info.hp_us, flows)
            earliest_deadline = min(
                (f.deadline_us for f in flows if f.deadline_us > 0), default=info.hp_us
            )
            queue_entries.append((spec_idx, spec.queue, required_us, float(earliest_deadline)))

        if not queue_entries:
            return {}

        raw_slots = [
            max(0.0, min(required_us / slot_us, float(self.n_slots)))
            for _, _, required_us, _ in queue_entries
        ]
        budget_slots = max(len(queue_entries), int(round(self.n_slots * self.slot_cap)))
        max_duty = max(1, min(self.n_slots // 2 if self.n_slots > 1 else 1, budget_slots))

        int_slots: List[int] = []
        frac_parts: List[float] = []
        for slots in raw_slots:
            base = min(max(1, math.floor(slots)), max_duty)
            int_slots.append(base)
            frac_parts.append(max(0.0, min(1.0, slots - math.floor(slots))))

        used = sum(int_slots)
        if used > budget_slots:
            while used > budget_slots:
                idx = max(range(len(int_slots)), key=lambda i: int_slots[i])
                if int_slots[idx] <= 1:
                    break
                int_slots[idx] -= 1
                used -= 1
        elif used < budget_slots:
            for _ in range(max(0, (budget_slots - used) // 4)):
                idx = max(range(len(frac_parts)), key=lambda i: frac_parts[i])
                if int_slots[idx] >= max_duty:
                    frac_parts[idx] = 0.0
                    continue
                int_slots[idx] += 1
                frac_parts[idx] = 0.0

        ordered = sorted(range(len(queue_entries)), key=lambda i: queue_entries[i][3])
        schedule: Dict[int, Tuple[int, int]] = {}
        current = (port_idx * 2 + 2) % self.n_slots if self.n_slots > 0 else 0
        for pos in ordered:
            spec_idx = queue_entries[pos][0]
            duty = max(1, min(int_slots[pos], self.n_slots))
            schedule[spec_idx] = (current % self.n_slots if self.n_slots > 0 else 0, duty)
            current = (current + duty + 1) % self.n_slots if self.n_slots > 0 else 0
        return schedule

    def build_schedule(self) -> List[Tuple[int, int]]:
        if not self.scenario.queue_meta:
            return []
        actions: List[Tuple[int, int]] = [(0, 1)] * len(self.scenario.queue_meta)
        for port_idx, indices in self.port_queue_indices.items():
            for spec_idx, pair in self._schedule_port(port_idx, indices).items():
                actions[spec_idx] = pair
        return actions


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EDF baseline TAS scheduler.")
    p.add_argument("--scenario-dir", default="scenarios", help="Directory of scenario folders.")
    p.add_argument("--scenario-pattern", default=DEFAULT_SCENARIO_PATTERN, help="Scenario glob.")
    p.add_argument("--n-slots", type=int, default=8, help="Number of discrete gate slots.")
    p.add_argument("--guard", type=float, default=0.10, help="Guard-band fraction added to open time.")
    p.add_argument("--util-scale", type=float, default=0.5, help="Scaling of the required open time.")
    p.add_argument("--slot-cap", type=float, default=0.6, help="Per-port slot budget fraction.")
    p.add_argument("--gcl", default=GCL_XLSX, help="Scratch GCL workbook path.")
    p.add_argument("--out", default="edf_results.xlsx", help="Results Excel output path.")
    p.add_argument("--reward-sr", type=float, default=3.0, help="Reward weight on success rate.")
    p.add_argument("--reward-ou", type=float, default=3.0, help="Reward weight on overall utilisation.")
    p.add_argument("--reward-delay", type=float, default=0.15, help="Reward penalty weight on delay.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if not Path(args.scenario_dir).exists():
        print(f"Scenario directory '{args.scenario_dir}' not found.")
        sys.exit(1)
    scenario_paths = [s.dir for s in discover_scenarios(args.scenario_dir, args.scenario_pattern)]
    if not scenario_paths:
        print(f"No scenarios matching '{args.scenario_pattern}' in '{args.scenario_dir}'.")
        sys.exit(1)

    results: List[Dict[str, float]] = []
    for scenario_path in scenario_paths:
        scn = TasScenario(str(scenario_path))
        scheduler = EDFScheduler(
            scn, n_slots=args.n_slots, guard=args.guard,
            util_scale=args.util_scale, slot_cap=args.slot_cap,
        )
        actions = scheduler.build_schedule()
        if actions:
            write_gcl_from_actions(scn.port_list, scn.queue_meta, actions, args.n_slots, args.gcl)
        else:
            with pd.ExcelWriter(args.gcl, engine="xlsxwriter") as writer:
                pd.DataFrame(columns=["offset", "durations", "queueIndex"]).to_excel(
                    writer, index=False, sheet_name="Empty"
                )

        metrics = run_simulation(scenario_dir=str(scenario_path), gcl_path=args.gcl)
        sr = float(metrics["success_rate"])
        avg_ms = float(metrics["avg_delay_ms"])
        ou = float(metrics["ou_percent"])
        reward = reward_from_metrics(sr, ou, avg_ms, args.reward_sr, args.reward_ou, args.reward_delay)
        print(f"[EDF] {scn.name}: SR={sr*100:5.2f}% OU={ou:5.2f}% "
              f"AvgDelay={avg_ms:6.3f}ms Reward={reward:7.3f}")
        results.append({
            "scenario": scn.name,
            "success_rate": sr,
            "avg_delay_ms": avg_ms,
            "ou_percent": ou,
            "reward": reward,
        })

    pd.DataFrame(results).to_excel(args.out, index=False)
    print(f"[EDF] Metrics written to '{args.out}'.")


if __name__ == "__main__":
    main()
