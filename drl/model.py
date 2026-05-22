from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

K_PHASES = 5
P_CLASSES = 8
ACTION_DIM = K_PHASES + K_PHASES * P_CLASSES


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        msg_in = torch.cat([node_states[src], edge_attr], dim=-1)
        msg = self.dropout(self.msg_mlp(msg_in))
        agg = node_states.new_zeros(node_states.shape)
        agg.index_add_(0, dst, msg)
        out = torch.relu(self.self_proj(node_states) + agg)
        return self.norm(out)


class DeepGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int = 128,
        out_dim: int = 96,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [EdgeMessageLayer(hidden, edge_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.out_dim = out_dim
        self.hidden_dim = hidden
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(
        self, X: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_proj(X)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        h = self.norm(h)
        return self.out_proj(h)

    def export_config(self) -> Dict[str, float]:
        return {
            "node_dim": self.input_proj.in_features,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class TemplatePolicy(nn.Module):
    def __init__(
        self,
        node_dim: int,
        queue_dim: int,
        stats_dim: int,
        hidden: int = 128,
        k_phases: int = K_PHASES,
        p_classes: int = P_CLASSES,
    ):
        super().__init__()
        self.k_phases = k_phases
        self.p_classes = p_classes
        self.action_dim = k_phases + k_phases * p_classes

        self.queue_mlp = nn.Sequential(
            nn.Linear(node_dim + queue_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.stats_mlp = nn.Sequential(nn.Linear(stats_dim, 64), nn.ReLU())
        self.film = nn.Linear(64, 2 * hidden)
        self.attn_vec = nn.Parameter(torch.randn(hidden))

        self.tau_head = nn.Linear(hidden, k_phases)
        self.b_head = nn.Linear(hidden, k_phases * p_classes)
        self.value_head = nn.Linear(hidden, 1)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(
        self,
        queue_emb: torch.Tensor,
        queue_feat: torch.Tensor,
        stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.cat([queue_emb, queue_feat], dim=-1)
        h = self.queue_mlp(base)
        stats_enc = self.stats_mlp(stats).unsqueeze(0)
        gamma, beta = torch.chunk(self.film(stats_enc), 2, dim=-1)
        h = h * (1 + gamma) + beta
        attn_weights = torch.softmax(torch.matmul(h, self.attn_vec), dim=0)
        global_emb = torch.sum(attn_weights.unsqueeze(-1) * h, dim=0)

        tau_raw = self.tau_head(global_emb)
        b_raw = self.b_head(global_emb)
        action_mean = torch.cat([tau_raw, b_raw], dim=-1)
        value = self.value_head(global_emb).squeeze(-1)
        return action_mean, self.log_std, value
