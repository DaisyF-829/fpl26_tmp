"""
HeteroTimingMPNN：4 类有向边各自 edge_enc / edge_mlp；节点回归 + 全图 max-pool 后 graph_head 预测 cpd。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_max_pool
from torch_scatter import scatter

NUM_EDGE_TYPES = 4
NODE_KEY = "tnode"


def _rel(k: int) -> tuple[str, str, str]:
    return (NODE_KEY, f"e{k}", NODE_KEY)


class HeteroMPNNLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        h = hidden_dim
        self.edge_mlps = nn.ModuleDict(
            {
                str(k): nn.Sequential(
                    nn.Linear(3 * h, 2 * h),
                    nn.ReLU(inplace=True),
                    nn.Linear(2 * h, h),
                )
                for k in range(NUM_EDGE_TYPES)
            }
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * h, 2 * h),
            nn.ReLU(inplace=True),
            nn.Linear(2 * h, h),
        )
        self.norm = nn.LayerNorm(h)

    def forward(
        self,
        h: torch.Tensor,
        hetero: HeteroData,
        edge_embs: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        agg = torch.zeros_like(h)
        for k in range(NUM_EDGE_TYPES):
            rel = _rel(k)
            ei = hetero[rel].edge_index
            if ei.numel() == 0:
                continue
            src, dst = ei[0], ei[1]
            ee = edge_embs[k]
            m_in = torch.cat([h[src], h[dst], ee], dim=-1)
            m = self.edge_mlps[str(k)](m_in)
            agg = agg + scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")
        h_upd = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_upd)


class HeteroTimingMPNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()
        h = hidden_dim
        self.node_enc = nn.Sequential(
            nn.Linear(14, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h),
        )
        self.edge_encs = nn.ModuleDict(
            {
                str(k): nn.Sequential(
                    nn.Linear(8, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, h),
                )
                for k in range(NUM_EDGE_TYPES)
            }
        )
        self.layers = nn.ModuleList([HeteroMPNNLayer(h) for _ in range(num_layers)])
        self.reg_head = nn.Sequential(
            nn.Linear(h, max(h // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(h // 2, 1), 1),
        )
        self.graph_head = nn.Sequential(
            nn.Linear(h, max(h // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(h // 2, 1), 1),
        )

    def forward(self, batch: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_enc(batch[NODE_KEY].x)
        edge_embs: dict[int, torch.Tensor] = {}
        for k in range(NUM_EDGE_TYPES):
            rel = _rel(k)
            edge_embs[k] = self.edge_encs[str(k)](batch[rel].edge_attr)
        for layer in self.layers:
            h = layer(h, batch, edge_embs)
        node_pred = self.reg_head(h).squeeze(-1)
        batch_vec = getattr(batch[NODE_KEY], "batch", None)
        g = global_max_pool(h, batch_vec)
        graph_pred = self.graph_head(g).squeeze(-1)
        return node_pred, graph_pred
