from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from drl.environment import (
    RolloutSample,
    TasScenario,
    compile_gcl_from_template,
    reward_from_metrics,
)
from drl.model import DeepGNN, K_PHASES, P_CLASSES, TemplatePolicy
from scenario_io import DEFAULT_SCENARIO_PATTERN, discover_scenarios
from sim.simulator import GCL_XLSX, run_simulation


class Trainer:
    def __init__(
        self,
        scenario_dir: str = "scenarios",
        scenario_pattern: str = DEFAULT_SCENARIO_PATTERN,
        k_phases: int = K_PHASES,
        p_classes: int = P_CLASSES,
        iters: int = 80,
        batch_episodes: int = 12,
        ppo_epochs: int = 3,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef_start: float = 0.01,
        ent_coef_end: float = 0.001,
        lr: float = 3e-4,
        feature_noise: float = 0.03,
        eval_every: int = 4,
        val_ratio: float = 0.2,
        w_sr: float = 3.0,
        w_ou: float = 3.0,
        w_delay: float = 0.15,
        device: str = "cpu",
        gcl_path: str = GCL_XLSX,
        model_best: str = "ppo_gnn_best.pth",
        model_last: str = "ppo_gnn_last.pth",
        log_path: str = "training_log.jsonl",
    ):
        self.device = torch.device(device)
        self.k_phases = k_phases
        self.p_classes = p_classes
        self.iters = iters
        self.batch_episodes = batch_episodes
        self.ppo_epochs = max(1, ppo_epochs)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.feature_noise_std = feature_noise
        self.eval_every = eval_every
        self.w_sr = w_sr
        self.w_ou = w_ou
        self.w_delay = w_delay
        self.gcl_path = gcl_path
        self.model_best = model_best
        self.model_last = model_last

        scenario_entries = discover_scenarios(scenario_dir, scenario_pattern)
        if not scenario_entries:
            raise ValueError(f"No scenarios matching '{scenario_pattern}' in '{scenario_dir}'.")
        self.scenarios = [TasScenario(str(entry.dir)) for entry in scenario_entries]
        self.scenario_count = len(self.scenarios)

        self.topology_sigs = sorted({scn.topology_sig for scn in self.scenarios})
        self.topology_sig = self.topology_sigs[0] if self.topology_sigs else ""
        if len(self.topology_sigs) > 1:
            print(
                f"[INFO] Training across {len(self.topology_sigs)} topology variants; "
                "the policy learns to generalise over them."
            )
        if not self.scenarios[0].port_list:
            raise ValueError("Scenarios must contain at least one switch port.")

        self.train_scenarios, self.val_scenarios = self._split_scenarios(self.scenarios, val_ratio)
        self._train_order: List[int] = list(range(len(self.train_scenarios)))
        self._cursor = 0
        random.shuffle(self._train_order)

        template = self.scenarios[0]
        self.gnn = DeepGNN(
            node_dim=template.X_base.shape[1],
            edge_dim=template.edge_attr.shape[1],
            hidden=128,
            out_dim=96,
            num_layers=3,
            dropout=0.1,
        ).to(self.device)
        self.pi = TemplatePolicy(
            node_dim=self.gnn.out_dim,
            queue_dim=template.queue_features.shape[1],
            stats_dim=template.scenario_stats.shape[0],
            hidden=128,
            k_phases=k_phases,
            p_classes=p_classes,
        ).to(self.device)
        self.opt = optim.Adam(
            list(self.gnn.parameters()) + list(self.pi.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
        )

        self.best_val_reward = -float("inf")
        self._log_path = log_path
        self._log_file = open(self._log_path, "w", encoding="utf-8")

    def _log(self, record: dict) -> None:
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    @staticmethod
    def _split_scenarios(
        scenarios: List[TasScenario], val_ratio: float
    ) -> Tuple[List[TasScenario], List[TasScenario]]:
        idx = list(range(len(scenarios)))
        random.Random(1234).shuffle(idx)
        val_count = max(1, int(len(idx) * val_ratio))
        val_idx = idx[:val_count]
        train_idx = idx[val_count:] or idx[:1]
        return [scenarios[i] for i in train_idx], [scenarios[i] for i in val_idx]

    def entropy_coef(self, step: int, total_steps: int) -> float:
        t = step / max(1, total_steps - 1)
        return (1 - t) * self.ent_coef_start + t * self.ent_coef_end

    def _next_scenario(self) -> TasScenario:
        if self._cursor >= len(self._train_order):
            random.shuffle(self._train_order)
            self._cursor = 0
        scn = self.train_scenarios[self._train_order[self._cursor]]
        self._cursor += 1
        return scn

    def _prepare_inputs(
        self, scn: TasScenario, noise: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, X, queue_feat, stats = scn.tensors(noise)
        return (
            scn.edge_index.to(self.device),
            scn.edge_attr.to(self.device),
            X.to(self.device),
            queue_feat.to(self.device),
            stats.to(self.device),
        )

    def sample_actions(self, scn: TasScenario) -> RolloutSample:
        edge_index, edge_attr, X, queue_feat, stats = self._prepare_inputs(scn, self.feature_noise_std)
        queue_idx = scn.queue_node_idx.to(self.device)
        node_emb = self.gnn(X, edge_index, edge_attr)
        queue_emb = node_emb[queue_idx]

        action_mean, log_std, value = self.pi(queue_emb, queue_feat, stats)
        std = torch.exp(log_std.clamp(-5, 2))
        dist = torch.distributions.Normal(action_mean, std)
        action_raw = dist.sample()
        logp = dist.log_prob(action_raw).sum()
        entropy = dist.entropy().sum()

        tau = torch.softmax(action_raw[: self.k_phases], dim=0)
        B = torch.sigmoid(action_raw[self.k_phases :]).reshape(self.k_phases, self.p_classes)

        return RolloutSample(
            edge_index=edge_index.detach(),
            edge_attr=edge_attr.detach(),
            node_feat=X.detach(),
            queue_feat=queue_feat.detach(),
            stats=stats.detach(),
            queue_idx=queue_idx.detach(),
            action_raw=action_raw.detach(),
            tau=tau.detach(),
            B=B.detach(),
            logp_old=logp.detach(),
            value_old=value.detach(),
            entropy=entropy.detach(),
        )

    @torch.no_grad()
    def greedy_actions(self, scn: TasScenario) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index, edge_attr, X, queue_feat, stats = self._prepare_inputs(scn, noise=0.0)
        node_emb = self.gnn(X, edge_index, edge_attr)
        queue_emb = node_emb[scn.queue_node_idx.to(self.device)]
        action_mean, _, _ = self.pi(queue_emb, queue_feat, stats)
        tau = torch.softmax(action_mean[: self.k_phases], dim=0)
        B = torch.sigmoid(action_mean[self.k_phases :]).reshape(self.k_phases, self.p_classes)
        return tau, B

    def evaluate_reward(self, scn: TasScenario) -> Tuple[float, float, float, float]:
        res = run_simulation(scenario_dir=scn.scenario_dir, gcl_path=self.gcl_path)
        sr = float(res["success_rate"])
        ou = float(res["ou_percent"])
        avg_ms = float(res["avg_delay_ms"])
        reward = reward_from_metrics(sr, ou, avg_ms, self.w_sr, self.w_ou, self.w_delay)
        return reward, sr, ou, avg_ms

    def evaluate_subset(self, scenarios: List[TasScenario], tag: str, update: int = 0) -> float:
        total = 0.0
        for scn in scenarios:
            tau, B = self.greedy_actions(scn)
            compile_gcl_from_template(tau, B, scn.port_list, scn.queue_meta, self.gcl_path)
            reward, sr, ou, ms = self.evaluate_reward(scn)
            total += reward
            print(f"[{tag}] {scn.name:>14s} SR={sr*100:5.1f}% OU={ou:5.1f}% Delay={ms:6.3f}ms R={reward:7.3f}")
        avg = total / max(1, len(scenarios))
        print(f"[{tag}] average reward = {avg:.3f}")
        self._log({"type": "validation", "update": update, "tag": tag, "avg_reward": avg,
                   "count": len(scenarios)})
        return avg

    def _scenario_metadata(self) -> List[Dict[str, object]]:
        return [
            {"path": str(Path(scn.scenario_dir)), "name": scn.name, "topology_sig": scn.topology_sig}
            for scn in self.scenarios
        ]

    def _checkpoint(self, path: str) -> None:
        torch.save(
            {
                "gnn": self.gnn.state_dict(),
                "pi": self.pi.state_dict(),
                "k_phases": self.k_phases,
                "p_classes": self.p_classes,
                "node_names": self.scenarios[0].node_names,
                "topology_sig": self.topology_sig,
                "topology_sigs": self.topology_sigs,
                "scenario_meta": self._scenario_metadata(),
                "gnn_config": self.gnn.export_config(),
                "policy_type": "template",
            },
            path,
        )

    def train(self) -> None:
        total_updates = self.iters
        for upd in range(1, total_updates + 1):
            ent_coef = self.entropy_coef(upd - 1, total_updates)
            batch_samples: List[RolloutSample] = []
            t0 = time.time()

            for b in range(self.batch_episodes):
                scn = self._next_scenario()
                if not scn.queue_meta:
                    continue
                sample = self.sample_actions(scn)
                compile_gcl_from_template(sample.tau, sample.B, scn.port_list, scn.queue_meta, self.gcl_path)
                reward, sr, ou, avg_ms = self.evaluate_reward(scn)
                sample.reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                batch_samples.append(sample)

                print(
                    f"[EP {upd:03d}.{b+1:02d}] {scn.name:>14s} SR={sr*100:5.1f}% "
                    f"OU={ou:5.1f}% Delay={avg_ms:6.3f}ms R={reward:7.3f}"
                )
                self._log({
                    "type": "episode", "update": upd, "episode": b + 1, "scenario": scn.name,
                    "success_rate": sr, "ou_percent": ou, "avg_delay_ms": avg_ms, "reward": reward,
                })

            if not batch_samples:
                continue

            rets = torch.stack([s.reward for s in batch_samples])
            values_old = torch.stack([s.value_old for s in batch_samples]).unsqueeze(-1)
            adv = rets - values_old
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).detach()
            rets = rets.detach()

            last = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
            for _ in range(self.ppo_epochs):
                policy_terms, value_terms, entropy_terms = [], [], []
                for idx, sample in enumerate(batch_samples):
                    node_emb = self.gnn(sample.node_feat, sample.edge_index, sample.edge_attr)
                    queue_emb = node_emb[sample.queue_idx]
                    action_mean, log_std, value = self.pi(queue_emb, sample.queue_feat, sample.stats)
                    std = torch.exp(log_std.clamp(-5, 2))
                    dist = torch.distributions.Normal(action_mean, std)
                    logp = dist.log_prob(sample.action_raw).sum()
                    ratio = torch.exp(logp - sample.logp_old)
                    surr1 = ratio * adv[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[idx]
                    policy_terms.append(-torch.min(surr1, surr2))
                    value_terms.append((rets[idx] - value).pow(2))
                    entropy_terms.append(dist.entropy().sum())

                policy_loss = torch.stack(policy_terms).mean()
                value_loss = self.vf_coef * torch.stack(value_terms).mean()
                entropy = torch.stack(entropy_terms).mean()
                loss = policy_loss + value_loss - ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.gnn.parameters()) + list(self.pi.parameters()), 0.5
                )
                self.opt.step()
                last = {
                    "policy": policy_loss.item(), "value": value_loss.item(),
                    "entropy": entropy.item(), "total": loss.item(),
                }

            dt = time.time() - t0
            print(
                f"[UPD {upd:03d}] loss={last['total']:.4f} pol={last['policy']:.4f} "
                f"val={last['value']:.4f} ent={last['entropy']:.4f} eta={ent_coef:.4f} "
                f"batch_time={dt:.2f}s"
            )
            self._log({
                "type": "update", "update": upd, "total_loss": last["total"],
                "policy_loss": last["policy"], "value_loss": last["value"],
                "entropy": last["entropy"], "entropy_coef": ent_coef, "batch_time_s": round(dt, 3),
            })

            if upd % self.eval_every == 0:
                val_reward = self.evaluate_subset(self.val_scenarios, "VAL", update=upd)
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self._checkpoint(self.model_best)
                    print(f"[CHECKPOINT] Best validation reward {val_reward:.3f} -> {self.model_best}")

        self._checkpoint(self.model_last)
        print(f"[SAVE] last model -> {self.model_last}")
        self._log_file.close()
        print(f"[LOG] training log -> {self._log_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the GNN + PPO TAS scheduler.")
    p.add_argument("--scenario-dir", default="scenarios", help="Directory of scenario folders.")
    p.add_argument("--scenario-pattern", default=DEFAULT_SCENARIO_PATTERN,
                   help="Glob used to discover scenarios.")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of scenarios held out for validation.")
    p.add_argument("--iters", type=int, default=80, help="Number of PPO update iterations.")
    p.add_argument("--batch-episodes", type=int, default=12, help="Rollout episodes per update.")
    p.add_argument("--ppo-epochs", type=int, default=3, help="Optimisation epochs per update.")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping parameter.")
    p.add_argument("--vf-coef", type=float, default=0.5, help="Value-loss coefficient.")
    p.add_argument("--ent-coef-start", type=float, default=0.01, help="Initial entropy coefficient.")
    p.add_argument("--ent-coef-end", type=float, default=0.001, help="Final entropy coefficient.")
    p.add_argument("--feature-noise", type=float, default=0.03, help="Std of feature noise during rollouts.")
    p.add_argument("--eval-every", type=int, default=4, help="Validate every N updates.")
    p.add_argument("--k-phases", type=int, default=K_PHASES, help="Number of TAS cycle phases.")
    p.add_argument("--p-classes", type=int, default=P_CLASSES, help="Number of traffic classes.")
    p.add_argument("--reward-sr", type=float, default=3.0, help="Reward weight on success rate.")
    p.add_argument("--reward-ou", type=float, default=3.0, help="Reward weight on overall utilisation.")
    p.add_argument("--reward-delay", type=float, default=0.15, help="Reward penalty weight on delay (ms).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--device", default="cpu", help="Torch device (cpu or cuda).")
    p.add_argument("--gcl", default=GCL_XLSX, help="Scratch GCL workbook path.")
    p.add_argument("--model-best", default="ppo_gnn_best.pth", help="Best-checkpoint output path.")
    p.add_argument("--model-last", default="ppo_gnn_last.pth", help="Final-checkpoint output path.")
    p.add_argument("--log", default="training_log.jsonl", help="JSONL training-log output path.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Trainer(
        scenario_dir=args.scenario_dir,
        scenario_pattern=args.scenario_pattern,
        k_phases=args.k_phases,
        p_classes=args.p_classes,
        iters=args.iters,
        batch_episodes=args.batch_episodes,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef_start=args.ent_coef_start,
        ent_coef_end=args.ent_coef_end,
        lr=args.lr,
        feature_noise=args.feature_noise,
        eval_every=args.eval_every,
        val_ratio=args.val_ratio,
        w_sr=args.reward_sr,
        w_ou=args.reward_ou,
        w_delay=args.reward_delay,
        device=args.device,
        gcl_path=args.gcl,
        model_best=args.model_best,
        model_last=args.model_last,
        log_path=args.log,
    ).train()


if __name__ == "__main__":
    main()
