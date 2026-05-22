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


def random_actions(scenario: TasScenario, n_slots: int) -> List[Tuple[int, int]]:
    return [
        (random.randrange(n_slots), random.randint(1, n_slots))
        for _ in scenario.queue_meta
    ]


def always_open_actions(scenario: TasScenario, n_slots: int) -> List[Tuple[int, int]]:
    return [(0, n_slots) for _ in scenario.queue_meta]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Naive (random / always-open) baseline schedulers.")
    p.add_argument("--mode", choices=["random", "always-open"], default="random",
                   help="Which naive scheduler to run.")
    p.add_argument("--scenario-dir", default="scenarios", help="Directory of scenario folders.")
    p.add_argument("--scenario-pattern", default=DEFAULT_SCENARIO_PATTERN, help="Scenario glob.")
    p.add_argument("--n-slots", type=int, default=8, help="Number of discrete gate slots.")
    p.add_argument("--num-seeds", type=int, default=30, help="Seeds averaged per scenario (random mode).")
    p.add_argument("--base-seed", type=int, default=42, help="First random seed (random mode).")
    p.add_argument("--gcl", default=GCL_XLSX, help="Scratch GCL workbook path.")
    p.add_argument("--out", default=None, help="Results Excel output path (default: <mode>_results.xlsx).")
    p.add_argument("--reward-sr", type=float, default=3.0, help="Reward weight on success rate.")
    p.add_argument("--reward-ou", type=float, default=3.0, help="Reward weight on overall utilisation.")
    p.add_argument("--reward-delay", type=float, default=0.15, help="Reward penalty weight on delay.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_path = args.out or f"{args.mode.replace('-', '_')}_results.xlsx"
    seeds = args.num_seeds if args.mode == "random" else 1

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
        for seed_offset in range(seeds):
            random.seed(args.base_seed + seed_offset)
            if args.mode == "random":
                actions = random_actions(scn, args.n_slots)
            else:
                actions = always_open_actions(scn, args.n_slots)
            write_gcl_from_actions(scn.port_list, scn.queue_meta, actions, args.n_slots, args.gcl)
            metrics = run_simulation(scenario_dir=str(scenario_path), gcl_path=args.gcl)
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
        print(f"[{args.mode.upper()}] {scn.name}: SR={avg_sr*100:5.2f}% OU={avg_ou:5.2f}% "
              f"AvgDelay={avg_ms:6.3f}ms Reward={reward:7.3f}")
        results.append({
            "scenario_index": scenario_index(scn.name) or len(results) + 1,
            "scenario": scn.name,
            "success_rate": avg_sr,
            "avg_delay_ms": avg_ms,
            "ou_percent": avg_ou,
            "reward": reward,
        })

    df = pd.DataFrame(results).sort_values("scenario_index").drop(columns=["scenario_index"])
    df.to_excel(out_path, index=False)
    print(f"[{args.mode.upper()}] Metrics written to '{out_path}'.")


if __name__ == "__main__":
    main()
