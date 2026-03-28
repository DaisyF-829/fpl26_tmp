"""
TimingMPNN：有向逐层消息传递（仅沿 src→dst，不加反向边）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter


class MPNNLayer(nn.Module):
    """单层有向消息传递：m_ij = MLP([h_src, h_dst, e_ij])，向 dst 求和聚合，残差 + LayerNorm。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        h = hidden_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * h, 2 * h),
            nn.ReLU(inplace=True),
            nn.Linear(2 * h, h),
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
        edge_index: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        m_in = torch.cat([h[src], h[dst], edge_emb], dim=-1)
        m = self.edge_mlp(m_in)
        agg = scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")
        h_upd = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_upd)


class TimingMPNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_enc = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim) for _ in range(num_layers)])
        h = hidden_dim
        self.reg_head = nn.Sequential(
            nn.Linear(h, max(h // 2, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(h // 2, 1), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_enc(x)
        edge_emb = self.edge_enc(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, edge_emb)
        out = self.reg_head(h).squeeze(-1)
        return out
