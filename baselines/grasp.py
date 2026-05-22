from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from drl.environment import TasScenario, reward_from_metrics, write_gcl_from_actions
from scenario_io import DEFAULT_SCENARIO_PATTERN, discover_scenarios, scenario_index
from sim.simulator import GCL_XLSX, run_simulation


class GRASPScheduler:
    def __init__(
        self,
        scenario: TasScenario,
        n_slots: int = 8,
        iters: int = 30,
        alpha: float = 0.45,
        local_iters: int = 15,
        target_scale: float = 0.7,
        jitter_strength: float = 0.05,
    ) -> None:
        self.scenario = scenario
        self.n_slots = n_slots
        self.iters = iters
        self.alpha = alpha
        self.local_iters = local_iters
        self.port_list = scenario.port_list
        self.queue_meta = scenario.queue_meta
        self.port_load = self._compute_port_loads()
        self.max_load = max(self.port_load.values(), default=1)
        self.target_scale = max(0.2, min(1.0, target_scale))
        self.jitter_strength = max(0.0, jitter_strength)

    def _compute_port_loads(self) -> Dict[Tuple[str, int], int]:
        counts: Dict[Tuple[str, int], int] = {}
        for _name, src, dst, _period, _queue in self.scenario.streams:
            path = self.scenario.paths.get((src, dst)) or []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = self.scenario.G[u].get(v)
                if edge is None:
                    continue
                key = (u, int(edge.get("portA", 0)))
                counts[key] = counts.get(key, 0) + 1
        return counts

    def _target_fraction(self, key: Tuple[str, int]) -> float:
        load = self.port_load.get(key, 1)
        scaled = self.target_scale * (load / max(1, self.max_load))
        return min(1.0, 0.05 + scaled)

    def _score(self, actions: List[Tuple[int, int]]) -> float:
        total = 0.0
        offset_hist: Dict[int, int] = {}
        for (offset, duty), spec in zip(actions, self.queue_meta):
            frac = duty / self.n_slots
            port = self.port_list[spec.port_idx]
            target = self._target_fraction((port.node, port.port))
            total += abs(frac - target)
            if frac > 0.6:
                total += 0.05 * (frac - 0.6)
            offset_hist[offset] = offset_hist.get(offset, 0) + 1
        total += 0.03 * sum(max(0, c - 1) for c in offset_hist.values())
        total += random.random() * self.jitter_strength
        return total

    def _construct(self) -> List[Tuple[int, int]]:
        actions: List[Tuple[int, int]] = []
        for q_idx, spec in enumerate(self.queue_meta):
            port = self.port_list[spec.port_idx]
            target = self._target_fraction((port.node, port.port))
            candidates: List[Tuple[float, int, int]] = []
            for duty in range(1, self.n_slots + 1):
                duty_pen = abs(duty / self.n_slots - target)
                for offset in range(self.n_slots):
                    sync_pen = 0.0
                    for prev_idx, (prev_off, _) in enumerate(actions):
                        if prev_off == offset:
                            sync_pen += 0.02
                        if prev_idx == q_idx - 1 and abs(prev_off - offset) <= 1:
                            sync_pen += 0.01
                    candidates.append((duty_pen + sync_pen + random.random() * 1e-3, offset, duty))
            candidates.sort(key=lambda x: x[0])
            threshold = candidates[0][0] + self.alpha * (candidates[-1][0] - candidates[0][0])
            rcl = [c for c in candidates if c[0] <= threshold]
            chosen = random.choice(rcl if rcl else candidates[:1])
            actions.append((chosen[1], chosen[2]))
        return actions

    def _local_search(self, actions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        best = list(actions)
        best_score = self._score(best)
        for _ in range(self.local_iters):
            idx = random.randrange(len(best))
            off, duty = best[idx]
            candidate = list(best)
            candidate[idx] = (
                (off + random.choice([-1, 0, 1])) % self.n_slots,
                max(1, min(self.n_slots, duty + random.choice([-1, 0, 1]))),
            )
            score = self._score(candidate)
            if score + 1e-6 < best_score:
                best, best_score = candidate, score
        return best

    def run(self) -> List[Tuple[int, int]]:
        if not self.queue_meta:
            return []
        best_actions: List[Tuple[int, int]] = []
        best_score = float("inf")
        for _ in range(max(1, self.iters)):
            actions = self._local_search(self._construct())
            score = self._score(actions)
            if score < best_score:
                best_score, best_actions = score, actions
        return best_actions


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRASP baseline TAS scheduler.")
    p.add_argument("--scenario-dir", default="scenarios", help="Directory of scenario folders.")
    p.add_argument("--scenario-pattern", default=DEFAULT_SCENARIO_PATTERN, help="Scenario glob.")
    p.add_argument("--iters", type=int, default=30, help="Global GRASP iterations (Long: 90).")
    p.add_argument("--local-iters", type=int, default=15, help="Local-search steps (Long: 45).")
    p.add_argument("--alpha", type=float, default=0.45, help="Restricted candidate-list greediness.")
    p.add_argument("--n-slots", type=int, default=8, help="Number of discrete gate slots.")
    p.add_argument("--num-seeds", type=int, default=30, help="Random seeds averaged per scenario.")
    p.add_argument("--base-seed", type=int, default=1234, help="First random seed.")
    p.add_argument("--gcl", default=GCL_XLSX, help="Scratch GCL workbook path.")
    p.add_argument("--out", default="grasp_results.xlsx", help="Results Excel output path.")
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
        runs: List[Dict[str, float]] = []
        for seed_offset in range(args.num_seeds):
            random.seed(args.base_seed + seed_offset)
            scheduler = GRASPScheduler(
                scn, n_slots=args.n_slots, iters=args.iters, alpha=args.alpha,
                local_iters=args.local_iters,
            )
            actions = scheduler.run()
            if actions:
                write_gcl_from_actions(scn.port_list, scn.queue_meta, actions, args.n_slots, args.gcl)
            else:
                with pd.ExcelWriter(args.gcl, engine="xlsxwriter"):
                    pass
            metrics = run_simulation(scenario_dir=str(scenario_path), gcl_path=args.gcl)
            runs.append({
                "success_rate": float(metrics["success_rate"]),
                "avg_delay_ms": float(metrics["avg_delay_ms"]),
                "ou_percent": float(metrics["ou_percent"]),
            })

        count = max(1, len(runs))
        avg_sr = sum(r["success_rate"] for r in runs) / count
        avg_delay = sum(r["avg_delay_ms"] for r in runs) / count
        avg_ou = sum(r["ou_percent"] for r in runs) / count
        reward = reward_from_metrics(avg_sr, avg_ou, avg_delay,
                                     args.reward_sr, args.reward_ou, args.reward_delay)
        print(f"[GRASP] {scn.name}: SR={avg_sr*100:5.2f}% OU={avg_ou:5.2f}% "
              f"AvgDelay={avg_delay:6.3f}ms Reward={reward:7.3f} (mean of {count} seeds)")
        results.append({
            "scenario_index": scenario_index(scn.name) or len(results) + 1,
            "scenario": scn.name,
            "success_rate": avg_sr,
            "avg_delay_ms": avg_delay,
            "ou_percent": avg_ou,
            "reward": reward,
            "seeds": count,
        })

    if results:
        df = pd.DataFrame(results).sort_values("scenario_index").drop(columns=["scenario_index"])
        df.to_excel(args.out, index=False)
        print(f"[GRASP] Metrics written to '{args.out}'.")


if __name__ == "__main__":
    main()
