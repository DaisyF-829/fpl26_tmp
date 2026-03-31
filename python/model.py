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

扩展（不在 2.4 节）: 另含全图 max-pooling + graph_head 预测归一化 CPD（目标为 1，与 rt_time/cpd 量纲一致），与节点损失加权求和（train.py）。
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


def _combined_edge_index(batch: HeteroData) -> torch.Tensor:
    """将 4 类异构边合并为一张同构有向图 edge_index（src->dst）。"""
    device = batch[NODE_KEY].x.device
    srcs: list[torch.Tensor] = []
    dsts: list[torch.Tensor] = []
    for k in range(NUM_EDGE_TYPES):
        rel = _rel(k)
        ei = batch[rel].edge_index
        if ei.numel() == 0:
            continue
        srcs.append(ei[0].to(device))
        dsts.append(ei[1].to(device))
    if not srcs:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return torch.stack([z, z], dim=0)
    return torch.stack([torch.cat(srcs, dim=0), torch.cat(dsts, dim=0)], dim=0)


def _k_hop_predecessor_agg(
    h: torch.Tensor, edge_index: torch.Tensor, k: int, *, reduce: str = "max"
) -> torch.Tensor:
    """
    沿有向边 src->dst 反复聚合前驱信息 k 次。
    返回对每个节点的 k-hop（按传播定义）前驱聚合表示，形状 [N, D]。
    """
    if k <= 0:
        return torch.zeros_like(h)
    if edge_index.numel() == 0:
        return torch.zeros_like(h)
    src, dst = edge_index[0], edge_index[1]
    x = h
    for _ in range(k):
        x = scatter(x[src], dst, dim=0, dim_size=h.size(0), reduce=reduce)
        # max 聚合时无入边的节点会得到 -inf；置 0 以保持数值稳定
        if reduce == "max":
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x


class _ResidualFuse(nn.Module):
    """用 MLP 融合 (self, agg)，并做残差 + LayerNorm。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        h = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * h, 2 * h),
            nn.ReLU(inplace=True),
            nn.Linear(2 * h, h),
        )
        self.norm = nn.LayerNorm(h)

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        delta = self.mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + delta)


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
                    nn.Linear(7, h),
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
        if batch_vec is None:
            batch_vec = h.new_zeros(h.size(0), dtype=torch.long)
        g = global_max_pool(h, batch_vec)
        graph_pred = self.graph_head(g).squeeze(-1)
        return node_pred, graph_pred


class HeteroTimingMPNNMultiHop(nn.Module):
    """
    在 1-hop 异构 MPNN 的基础上，额外收集 3/6/9 跳前驱聚合信息：
    - hop3/hop6/hop9 各自用一个独立可训练 MLP 融合 (h_self, h_hopK)
    - 每个融合块都带残差
    """

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
                    nn.Linear(7, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, h),
                )
                for k in range(NUM_EDGE_TYPES)
            }
        )
        self.layers = nn.ModuleList([HeteroMPNNLayer(h) for _ in range(num_layers)])

        self.fuse3 = _ResidualFuse(h)
        self.fuse6 = _ResidualFuse(h)
        self.fuse9 = _ResidualFuse(h)

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

        ei = _combined_edge_index(batch)
        hop3 = _k_hop_predecessor_agg(h, ei, 3, reduce="max")
        hop6 = _k_hop_predecessor_agg(h, ei, 6, reduce="max")
        hop9 = _k_hop_predecessor_agg(h, ei, 9, reduce="max")

        h = self.fuse3(h, hop3)
        h = self.fuse6(h, hop6)
        h = self.fuse9(h, hop9)

        node_pred = self.reg_head(h).squeeze(-1)
        batch_vec = getattr(batch[NODE_KEY], "batch", None)
        if batch_vec is None:
            batch_vec = h.new_zeros(h.size(0), dtype=torch.long)
        g = global_max_pool(h, batch_vec)
        graph_pred = self.graph_head(g).squeeze(-1)
        return node_pred, graph_pred


class HeteroTimingMPNNDelayProp(nn.Module):
    """
    两阶段架构：
    阶段1：num_layers 层 HeteroMPNNLayer，学习局部异构结构特征。
    阶段2：prop_steps 层 delay-weighted max 传播，用边的物理特征（7维）学一个标量权重
           w = sigmoid(Linear(7→1)(edge_attr))，固定权重，每步复用。
           每步：msg = h[src] * w → scatter_max → _ResidualFuse。
    标量 gate 只计算一次，速度与 _k_hop_predecessor_agg 相当，但引入了物理延迟先验。
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, prop_steps: int = 9):
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
                    nn.Linear(7, h),
                    nn.ReLU(inplace=True),
                    nn.Linear(h, h),
                )
                for k in range(NUM_EDGE_TYPES)
            }
        )
        self.layers = nn.ModuleList([HeteroMPNNLayer(h) for _ in range(num_layers)])

        # 标量 gate：edge_attr(7维) → 1个标量权重，所有 prop_steps 共享
        self.delay_gate = nn.Linear(7, 1)

        # 每步有独立的 fuse（保持各步可学习性），节点上计算，开销小
        self.prop_fuses = nn.ModuleList([_ResidualFuse(h) for _ in range(prop_steps)])

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

        # 阶段1：HeteroMPNN 局部传播
        for layer in self.layers:
            h = layer(h, batch, edge_embs)

        # 阶段2：delay-weighted max 传播
        # 合并 4 类边，预计算标量 gate（只算一次）
        ei = _combined_edge_index(batch)
        if ei.numel() > 0:
            all_eattr: list[torch.Tensor] = []
            for k in range(NUM_EDGE_TYPES):
                rel = _rel(k)
                ea = batch[rel].edge_attr
                if ea.size(0) > 0:
                    all_eattr.append(ea)
            if all_eattr:
                combined_eattr = torch.cat(all_eattr, dim=0)  # [E_total, 7]
                w = torch.sigmoid(self.delay_gate(combined_eattr))  # [E_total, 1]
            else:
                w = torch.ones(ei.size(1), 1, device=h.device)

            src, dst = ei[0], ei[1]
            for fuse in self.prop_fuses:
                msg = h[src] * w  # [E, h]，broadcast
                agg = scatter(msg, dst, dim=0, dim_size=h.size(0), reduce="max")
                agg = torch.where(torch.isfinite(agg), agg, torch.zeros_like(agg))
                h = fuse(h, agg)

        node_pred = self.reg_head(h).squeeze(-1)
        batch_vec = getattr(batch[NODE_KEY], "batch", None)
        if batch_vec is None:
            batch_vec = h.new_zeros(h.size(0), dtype=torch.long)
        g = global_max_pool(h, batch_vec)
        graph_pred = self.graph_head(g).squeeze(-1)
        return node_pred, graph_pred
