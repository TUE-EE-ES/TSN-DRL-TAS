from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from drl.environment import TasScenario, reward_from_metrics, write_gcl_from_actions
from scenario_io import DEFAULT_SCENARIO_PATTERN, discover_scenarios, scenario_index
from sim.simulator import GCL_XLSX, run_simulation


class TabuSearchScheduler:
    def __init__(
        self,
        scenario: TasScenario,
        n_slots: int = 8,
        iterations: int = 10,
        neighbor_samples: int = 3,
        tabu_tenure: int = 4,
        early_stop: int = 4,
        gcl_path: str = GCL_XLSX,
    ) -> None:
        self.scenario = scenario
        self.n_slots = n_slots
        self.iterations = iterations
        self.neighbor_samples = neighbor_samples
        self.tabu_tenure = tabu_tenure
        self.early_stop = early_stop
        self.gcl_path = gcl_path
        self.cache: Dict[Tuple[Tuple[int, int], ...], Dict[str, float]] = {}
        self.queue_meta = scenario.queue_meta
        self.queue_count = len(self.queue_meta)
        self.port_load = self._compute_port_loads()
        self.max_port_load = max(self.port_load.values(), default=1)
        self.estimate_bias = 0.15

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

    def _initial_actions(self) -> List[Tuple[int, int]]:
        actions: List[Tuple[int, int]] = [(0, 1)] * self.queue_count
        indices = list(range(self.queue_count))
        random.shuffle(indices)
        for idx in indices:
            spec = self.queue_meta[idx]
            port = self.scenario.port_list[spec.port_idx]
            load = self.port_load.get((port.node, port.port), 1)
            frac = min(1.0, max(0.1, load / max(1, self.max_port_load) + random.uniform(-0.1, 0.1)))
            duty = max(1, min(self.n_slots, int(round(frac * self.n_slots * 0.6)) or 1))
            duty = max(1, min(self.n_slots, duty + random.choice([-1, 0, 1])))
            base_offset = (idx * 3 + spec.queue) % self.n_slots
            offset = (base_offset + random.choice([-1, 0, 1])) % self.n_slots
            actions[idx] = (offset, duty)
        return actions

    def _estimate_reward(self, actions: Sequence[Tuple[int, int]]) -> float:
        score = 0.0
        for action, spec in zip(actions, self.queue_meta):
            port = self.scenario.port_list[spec.port_idx]
            load = self.port_load.get((port.node, port.port), 1)
            target = min(1.0, load / max(1, self.max_port_load))
            score -= abs(max(1, action[1]) / self.n_slots - target)
            score -= 0.05 * (action[0] / self.n_slots)
        return score - self.estimate_bias

    def _neighbor(self, base: Sequence[Tuple[int, int]], idx: int, d_off: int, d_duty: int):
        actions = list(base)
        off, duty = actions[idx]
        actions[idx] = (
            (off + d_off) % self.n_slots,
            max(1, min(self.n_slots, duty + d_duty)),
        )
        return actions

    def _evaluate(self, actions: Sequence[Tuple[int, int]]) -> Dict[str, float]:
        key = tuple(actions)
        if key in self.cache:
            return self.cache[key]
        write_gcl_from_actions(
            self.scenario.port_list, self.queue_meta, list(actions), self.n_slots, self.gcl_path
        )
        metrics = run_simulation(scenario_dir=self.scenario.scenario_dir, gcl_path=self.gcl_path)
        self.cache[key] = metrics
        return metrics

    def search(self) -> Dict[str, float]:
        if not self.queue_meta:
            return self._evaluate([(0, 1)])

        actions = self._initial_actions()
        best_actions = list(actions)
        best_estimate = self._estimate_reward(actions)
        no_improve = 0
        tabu_queue: Deque[Tuple[int, int, int]] = deque()
        tabu_set: set[Tuple[int, int, int]] = set()

        for _ in range(self.iterations):
            neighborhood = []
            attempts = 0
            while len(neighborhood) < self.neighbor_samples and attempts < self.neighbor_samples * 3:
                attempts += 1
                idx = random.randrange(len(actions))
                d_off = random.choice([-1, 0, 1])
                d_duty = random.choice([-1, 0, 1])
                if d_off == 0 and d_duty == 0:
                    continue
                move = (
                    idx,
                    (actions[idx][0] + d_off) % self.n_slots,
                    max(1, min(self.n_slots, actions[idx][1] + d_duty)),
                )
                candidate = self._neighbor(actions, idx, d_off, d_duty)
                est = self._estimate_reward(candidate)
                if move in tabu_set and est <= best_estimate:
                    continue
                neighborhood.append((est, candidate, move))

            if not neighborhood:
                break

            estimate, candidate, move = max(neighborhood, key=lambda item: item[0])
            actions = candidate
            if estimate > best_estimate:
                best_estimate, best_actions, no_improve = estimate, candidate, 0
            else:
                no_improve += 1

            tabu_queue.append(move)
            tabu_set.add(move)
            if len(tabu_queue) > self.tabu_tenure:
                tabu_set.discard(tabu_queue.popleft())
            if no_improve >= self.early_stop:
                break

        return self._evaluate(best_actions)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tabu Search baseline TAS scheduler.")
    p.add_argument("--scenario-dir", default="scenarios", help="Directory of scenario folders.")
    p.add_argument("--scenario-pattern", default=DEFAULT_SCENARIO_PATTERN, help="Scenario glob.")
    p.add_argument("--iterations", type=int, default=10, help="Search iterations (Long: 50).")
    p.add_argument("--neighbor-samples", type=int, default=3, help="Neighbours per iteration (Long: 30).")
    p.add_argument("--tabu-tenure", type=int, default=4, help="Tabu memory length (Long: 20).")
    p.add_argument("--early-stop", type=int, default=4, help="Stop after N non-improving steps (Long: 40).")
    p.add_argument("--n-slots", type=int, default=8, help="Number of discrete gate slots.")
    p.add_argument("--num-seeds", type=int, default=30, help="Random seeds averaged per scenario.")
    p.add_argument("--base-seed", type=int, default=42, help="First random seed.")
    p.add_argument("--gcl", default=GCL_XLSX, help="Scratch GCL workbook path.")
    p.add_argument("--out", default="tabu_results.xlsx", help="Results Excel output path.")
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
            scheduler = TabuSearchScheduler(
                scn, n_slots=args.n_slots, iterations=args.iterations,
                neighbor_samples=args.neighbor_samples, tabu_tenure=args.tabu_tenure,
                early_stop=args.early_stop, gcl_path=args.gcl,
            )
            metrics = scheduler.search()
            runs.append({
                "success_rate": float(metrics["success_rate"]),
                "avg_delay_ms": float(metrics["avg_delay_ms"]),
                "ou_percent": float(metrics["ou_percent"]),
            })

        count = max(1, len(runs))
        avg_sr = sum(r["success_rate"] for r in runs) / count
        avg_ms = sum(r["avg_delay_ms"] for r in runs) / count
        avg_ou = sum(r["ou_percent"] for r in runs) / count
        reward = reward_from_metrics(avg_sr, avg_ou, avg_ms,
                                     args.reward_sr, args.reward_ou, args.reward_delay)
        print(f"[TABU] {scn.name}: SR={avg_sr*100:5.2f}% OU={avg_ou:5.2f}% "
              f"AvgDelay={avg_ms:6.3f}ms Reward={reward:7.3f} (mean of {count} seeds)")
        results.append({
            "scenario_index": scenario_index(scn.name) or len(results) + 1,
            "scenario": scn.name,
            "success_rate": avg_sr,
            "avg_delay_ms": avg_ms,
            "ou_percent": avg_ou,
            "reward": reward,
            "seeds": count,
        })

    df = pd.DataFrame(results).sort_values("scenario_index").drop(columns=["scenario_index"])
    df.to_excel(args.out, index=False)
    print(f"[TABU] Metrics written to '{args.out}'.")


if __name__ == "__main__":
    main()
