"""
对照实验用「平凡」异构卷积栈：每层对 4 类边各一路 conv，HeteroConv 在关系间 sum；不使用边特征。

含 GCN / GAT / GraphSAGE / GIN，接口与 model.HeteroTimingMPNN 一致：forward(batch) -> (node_pred, graph_pred)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, GINConv, HeteroConv, SAGEConv, global_max_pool

from data_loader import NUM_EDGE_TYPES, NODE_KEY, edge_type_key


def _hetero_conv_forward(
    node_enc: nn.Module,
    convs: nn.ModuleList,
    norms: nn.ModuleList,
    reg_head: nn.Module,
    graph_head: nn.Module,
    batch: HeteroData,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = node_enc(batch[NODE_KEY].x)
    x_dict: dict[str, torch.Tensor] = {NODE_KEY: h}
    edge_index_dict = {
        edge_type_key(k): batch[edge_type_key(k)].edge_index for k in range(NUM_EDGE_TYPES)
    }
    for conv, norm in zip(convs, norms):
        x_dict = conv(x_dict, edge_index_dict)
        h = norm(F.relu(x_dict[NODE_KEY]))
        x_dict = {NODE_KEY: h}
    node_pred = reg_head(h).squeeze(-1)
    batch_vec = getattr(batch[NODE_KEY], "batch", None)
    if batch_vec is None:
        batch_vec = h.new_zeros(h.size(0), dtype=torch.long)
    g = global_max_pool(h, batch_vec)
    graph_pred = graph_head(g).squeeze(-1)
    return node_pred, graph_pred


class _HeteroTimingConvNet(nn.Module):
    """子类在 __init__ 中向 self.convs 追加若干 HeteroConv。"""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        h = hidden_dim
        self.node_enc = nn.Sequential(
            nn.Linear(14, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(num_layers)])
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
        return _hetero_conv_forward(
            self.node_enc, self.convs, self.norms, self.reg_head, self.graph_head, batch
        )


def _gin_mlp(dim: int) -> nn.Sequential:
    """标准 GIN 的 2 层 MLP（逐关系独立实例）。"""
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, dim),
    )


class HeteroTimingGCN(_HeteroTimingConvNet):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__(hidden_dim, num_layers)
        h = hidden_dim
        for _ in range(num_layers):
            conv_dict = {edge_type_key(k): GCNConv(h, h) for k in range(NUM_EDGE_TYPES)}
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))


class HeteroTimingGAT(_HeteroTimingConvNet):
    """单头 GAT，不拼接多 head（任意 hidden_dim 可用）。"""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6, heads: int = 1):
        super().__init__(hidden_dim, num_layers)
        h = hidden_dim
        for _ in range(num_layers):
            conv_dict = {
                edge_type_key(k): GATConv(h, h, heads=heads, concat=False, dropout=0.0)
                for k in range(NUM_EDGE_TYPES)
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))


class HeteroTimingGraphSAGE(_HeteroTimingConvNet):
    """GraphSAGE 式 mean 邻域聚合。"""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__(hidden_dim, num_layers)
        h = hidden_dim
        for _ in range(num_layers):
            conv_dict = {
                edge_type_key(k): SAGEConv(h, h, aggr="mean") for k in range(NUM_EDGE_TYPES)
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))


class HeteroTimingGIN(_HeteroTimingConvNet):
    """GINConv：每类边独立 MLP + eps 可学习（PyG 默认）。"""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 6):
        super().__init__(hidden_dim, num_layers)
        h = hidden_dim
        for _ in range(num_layers):
            conv_dict = {
                edge_type_key(k): GINConv(_gin_mlp(h), train_eps=True) for k in range(NUM_EDGE_TYPES)
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))


# train / evaluate 用：CLI model_type -> (类, checkpoint 中的 model_class 名字符串)
HETERO_CONV_MODELS: dict[str, tuple[type[nn.Module], str]] = {
    "gcn": (HeteroTimingGCN, "HeteroTimingGCN"),
    "gat": (HeteroTimingGAT, "HeteroTimingGAT"),
    "sage": (HeteroTimingGraphSAGE, "HeteroTimingGraphSAGE"),
    "gin": (HeteroTimingGIN, "HeteroTimingGIN"),
}
