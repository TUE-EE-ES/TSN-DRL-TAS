from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import torch

from drl.environment import (
    TasScenario,
    compile_gcl_from_template,
    reward_from_metrics,
)
from drl.model import DeepGNN, K_PHASES, P_CLASSES, TemplatePolicy
from scenario_io import (
    DEFAULT_SCENARIO_PATTERN,
    discover_scenarios,
    resolve_scenario_dir,
)
from sim.simulator import GCL_XLSX, run_simulation

_SCENARIO_NUM = re.compile(r"(\d+)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained DRL TAS scheduler.")
    p.add_argument("--model", default="ppo_gnn_best.pth", help="Checkpoint file to load.")
    p.add_argument("--scenario-dir", help="Directory of scenario folders to evaluate.")
    p.add_argument("--scenario-pattern", default=DEFAULT_SCENARIO_PATTERN,
                   help="Glob used with --scenario-dir.")
    p.add_argument("--scenario", dest="scenarios", action="append",
                   help="Explicit scenario directory; repeat to add more.")
    p.add_argument("--skip-mismatch", action="store_true",
                   help="Skip scenarios whose topology differs from the checkpoint's.")
    p.add_argument("--device", default="cpu", help="Torch device (cpu or cuda).")
    p.add_argument("--gcl", default=GCL_XLSX, help="Scratch GCL workbook path.")
    p.add_argument("--out", default="drl_results.xlsx", help="Results Excel output path.")
    p.add_argument("--reward-sr", type=float, default=3.0, help="Reward weight on success rate.")
    p.add_argument("--reward-ou", type=float, default=3.0, help="Reward weight on overall utilisation.")
    p.add_argument("--reward-delay", type=float, default=0.15, help="Reward penalty weight on delay.")
    return p


def _sort_paths(paths: Iterable[Path]) -> List[Path]:
    def key(path: Path) -> tuple[int | float, str]:
        match = _SCENARIO_NUM.search(path.stem)
        return (int(match.group(1)) if match else float("inf"), path.stem)

    return sorted(paths, key=key)


def _resolve_meta_path(entry: dict, fallback_dir: str) -> Optional[Path]:
    candidate = Path(entry["path"])
    for option in (candidate, Path(candidate.name), Path(fallback_dir) / candidate.name):
        if option.exists():
            return option
    return None


def _gather_scenarios(args: argparse.Namespace, scenario_meta: Optional[Sequence[dict]]) -> List[Path]:
    if args.scenarios:
        return _sort_paths(resolve_scenario_dir(p).dir for p in args.scenarios)
    if args.scenario_dir:
        directory = Path(args.scenario_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Scenario directory '{directory}' does not exist.")
        return _sort_paths(s.dir for s in discover_scenarios(directory, args.scenario_pattern))
    if scenario_meta:
        resolved = [p for p in (_resolve_meta_path(e, "scenarios") for e in scenario_meta) if p]
        if resolved:
            return _sort_paths(resolved)
    raise ValueError(
        "No scenarios specified. Provide --scenario-dir or --scenario, or use a checkpoint "
        "that records its training scenarios."
    )


def _topology_ok(scn: TasScenario, ckpt: dict) -> bool:
    sigs = ckpt.get("topology_sigs")
    if sigs:
        return scn.topology_sig in sigs
    sig = ckpt.get("topology_sig")
    if sig:
        return scn.topology_sig == sig
    return scn.node_names == ckpt.get("node_names")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    ckpt = torch.load(args.model, map_location="cpu")
    scenario_meta = ckpt.get("scenario_meta")
    scenario_paths = _gather_scenarios(args, scenario_meta)

    k_phases = ckpt.get("k_phases", K_PHASES)
    p_classes = ckpt.get("p_classes", P_CLASSES)
    gnn_cfg = ckpt.get("gnn_config", {})

    template = TasScenario(str(scenario_paths[0]))
    gnn = DeepGNN(
        node_dim=template.X_base.shape[1],
        edge_dim=template.edge_attr.shape[1],
        hidden=int(gnn_cfg.get("hidden_dim", 128)),
        out_dim=int(gnn_cfg.get("out_dim", 96)),
        num_layers=int(gnn_cfg.get("num_layers", 3)),
        dropout=float(gnn_cfg.get("dropout", 0.1)),
    )
    gnn.load_state_dict(ckpt["gnn"])
    policy = TemplatePolicy(
        node_dim=gnn.out_dim,
        queue_dim=template.queue_features.shape[1],
        stats_dim=template.scenario_stats.shape[0],
        hidden=128,
        k_phases=k_phases,
        p_classes=p_classes,
    )
    policy.load_state_dict(ckpt["pi"])

    device = torch.device(args.device)
    gnn.to(device).eval()
    policy.to(device).eval()

    results: List[dict] = []
    skipped: List[str] = []

    for scenario_path in scenario_paths:
        scn = TasScenario(str(scenario_path))
        if not _topology_ok(scn, ckpt):
            msg = f"Topology mismatch for scenario '{scn.name}'."
            if args.skip_mismatch:
                print(f"[SKIP] {msg}")
                skipped.append(scn.name)
                continue
            raise ValueError(msg + " Use --skip-mismatch to ignore.")

        with torch.no_grad():
            _, X, queue_feat, stats = scn.tensors(noise_std=0.0)
            node_emb = gnn(X.to(device), scn.edge_index.to(device), scn.edge_attr.to(device))
            queue_emb = node_emb[scn.queue_node_idx.to(device)]
            action_mean, _, _ = policy(queue_emb, queue_feat.to(device), stats.to(device))
            tau = torch.softmax(action_mean[:k_phases], dim=0)
            B = torch.sigmoid(action_mean[k_phases:]).reshape(k_phases, p_classes)
            compile_gcl_from_template(tau, B, scn.port_list, scn.queue_meta, args.gcl)

        metrics = run_simulation(scenario_dir=str(scenario_path), gcl_path=args.gcl)
        sr = float(metrics["success_rate"])
        avg_ms = float(metrics["avg_delay_ms"])
        ou = float(metrics["ou_percent"])
        reward = reward_from_metrics(sr, ou, avg_ms, args.reward_sr, args.reward_ou, args.reward_delay)
        match = _SCENARIO_NUM.search(scn.name)
        print(f"[TEST] {scn.name}: SR={sr*100:5.2f}% OU={ou:5.2f}% AvgDelay={avg_ms:6.3f}ms Reward={reward:7.3f}")
        results.append({
            "scenario_index": int(match.group(1)) if match else len(results) + 1,
            "scenario": scn.name,
            "success_rate": sr,
            "avg_delay_ms": avg_ms,
            "ou_percent": ou,
            "reward": reward,
        })

    if results:
        df = pd.DataFrame(results).sort_values("scenario_index").drop(columns=["scenario_index"])
        df.to_excel(args.out, index=False)
        print(f"[TEST] Metrics written to '{args.out}'.")
    else:
        print("[TEST] No scenarios evaluated; nothing to write.")

    if skipped:
        print(f"[TEST] Skipped (topology mismatch): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
