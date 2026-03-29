"""
异构时序图 MPNN（与 STA 有向传播一致）

与文档 2.3.1–2.4 的对应关系
--------------------------
Step 1（逐类型消息）: 对边类型 k，先经 edge_enc 将原始边特征映射为 d 维 \\mathbf{e}_{uv}^{(k)}，
再 m^{(k)} = MLP_msg^{(k)}([h_u, h_v, e^{(k)}])，其中 MLP_msg 为 Linear(3d→2d)→ReLU→Linear(2d→d)。

Step 2（跨类型汇聚）: 对各类边在目标节点 v 上对消息求和，实现 a_v = Σ_k Σ_{u∈N^{(k)}(v)} m_{uv}^{(k)}
（torch_scatter.scatter(..., reduce="sum") 到 dst）。

Step 3（残差 + LayerNorm）: h^{(l+1)} = LayerNorm(h^{(l)} + MLP_upd([h^{(l)}, a^{(l)}]))，
MLP_upd 为 Linear(2d→2d)→ReLU→Linear(2d→d)。

2.3.2 传播方向: 仅沿有向边 u→v（edge_index 中 src→dst），不向反向边传播。

2.4 回归头: \\hat{y}_v = MLP_reg(h^{(L)})，MLP_reg 为 Linear(d→d/2)→ReLU→Linear(d/2→1)，
输出为归一化到达时间。训练时节点 MSE 仅在 y_valid 上计算（见 data_loader / train.py），
目标为 t_v/CPD。

扩展（不在 2.4 节）: 另含全图 max-pooling + graph_head 预测 CPD 标量，与节点损失加权求和（train.py）。
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
    """单层 2.3.1：类型相关 MLP_msg、跨类型 sum 聚合、MLP_upd + 残差 + LayerNorm。"""

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
    """L 层 HeteroMPNNLayer + 2.4 节 MLP_reg；另含 graph_head（全图辅助任务）。"""

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
