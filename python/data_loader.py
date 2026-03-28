"""
将 C++ 导出的 timing_graph.npz 转为 torch_geometric.data.Data。
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """indices: 整型 [N] 或 [E]，返回 float32 [N/E, num_classes]。"""
    idx = indices.astype(np.int64).clip(0, num_classes - 1)
    oh = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(idx.shape[0]), idx] = 1.0
    return oh


def _build_y_critical_from_top_paths(
    top_path_node_ids: np.ndarray | None, num_nodes: int, K: int = 20
) -> np.ndarray:
    """并集：前 K 条 top path 上出现过的节点为 True。"""
    mask = np.zeros(num_nodes, dtype=bool)
    if top_path_node_ids is None or top_path_node_ids.size == 0:
        return mask
    k_use = min(K, int(top_path_node_ids.shape[0]))
    for p in range(k_use):
        row = top_path_node_ids[p]
        for v in np.nditer(row):
            vi = int(v)
            if 0 <= vi < num_nodes:
                mask[vi] = True
    return mask


def load_timing_graph(npz_path: str) -> Data:
    z = np.load(npz_path, allow_pickle=False)

    def arr(name: str, default=None):
        if name not in z.files:
            return default
        return z[name]

    tnode_type = arr("tnode_type")
    if tnode_type is None:
        raise KeyError(f"{npz_path} 缺少 tnode_type")

    N = int(tnode_type.shape[0])
    cpd = float(arr("critical_path_delay", np.array([1.0], dtype=np.float32))[0])
    if cpd <= 0:
        cpd = 1.0

    grid_w = int(arr("grid_width", np.array([1], dtype=np.uint64))[0])
    grid_h = int(arr("grid_height", np.array([1], dtype=np.uint64))[0])
    grid_w = max(grid_w, 1)
    grid_h = max(grid_h, 1)
    gw_denom = max(grid_w - 1, 1)
    gh_denom = max(grid_h - 1, 1)
    grid_sum = float(grid_w + grid_h)

    pl_time = arr("tnode_pl_arrival", np.full(N, -1.0, dtype=np.float32)).astype(np.float32)
    pl_valid = arr("tnode_pl_arrival_mask", arr("tnode_pl_valid", np.zeros(N, dtype=np.int32)))
    pl_valid = pl_valid.astype(np.float32).reshape(-1)[:N]

    tnode_x = arr("tnode_x", np.zeros(N, dtype=np.int32)).astype(np.float32).reshape(-1)[:N]
    tnode_y = arr("tnode_y", np.zeros(N, dtype=np.int32)).astype(np.float32).reshape(-1)[:N]
    fanin = arr("tnode_fanin", np.zeros(N, dtype=np.int32)).astype(np.float32).reshape(-1)[:N]
    fanout = arr("tnode_fanout", np.zeros(N, dtype=np.int32)).astype(np.float32).reshape(-1)[:N]
    topo = arr("tnode_topo_level", np.full(N, -1, dtype=np.int32)).astype(np.float32).reshape(-1)[:N]
    net_hpwl_n = arr("tnode_net_hpwl", np.full(N, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:N]
    net_fanout_n = arr("tnode_net_fanout", np.full(N, -1, dtype=np.int32)).astype(np.float32).reshape(-1)[:N]

    valid_topo = topo >= 0
    max_level = float(topo[valid_topo].max()) if valid_topo.any() else 0.0
    if max_level <= 0:
        max_level = 1.0
    topo_norm = np.where(valid_topo, topo / max_level, 0.0).astype(np.float32)

    pl_feat = np.where(pl_valid > 0, pl_time / cpd, 0.0).astype(np.float32)

    x_coord = np.where(tnode_x >= 0, tnode_x / float(gw_denom), -0.05).astype(np.float32)
    y_coord = np.where(tnode_y >= 0, tnode_y / float(gh_denom), -0.05).astype(np.float32)

    hpwl_n = np.where(net_hpwl_n >= 0, net_hpwl_n / grid_sum, 0.0).astype(np.float32)
    nf_n = np.where(net_fanout_n >= 0, np.log1p(np.maximum(net_fanout_n, 0)), 0.0).astype(np.float32)

    type_oh = _one_hot(tnode_type.astype(np.int64), 5)
    x_np = np.concatenate(
        [
            type_oh,
            pl_feat.reshape(-1, 1),
            pl_valid.reshape(-1, 1),
            x_coord.reshape(-1, 1),
            y_coord.reshape(-1, 1),
            np.log1p(np.maximum(fanin, 0)).reshape(-1, 1),
            np.log1p(np.maximum(fanout, 0)).reshape(-1, 1),
            topo_norm.reshape(-1, 1),
            hpwl_n.reshape(-1, 1),
            nf_n.reshape(-1, 1),
        ],
        axis=1,
    )
    assert x_np.shape[1] == 14

    tedge_src = arr("tedge_src")
    tedge_dst = arr("tedge_dst")
    if tedge_src is None or tedge_dst is None:
        raise KeyError(f"{npz_path} 缺少 tedge_src / tedge_dst")
    E = int(tedge_src.shape[0])
    tedge_type = arr("tedge_type", np.zeros(E, dtype=np.uint64)).astype(np.int64).reshape(-1)[:E]
    tedge_delay = arr("tedge_delay", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    manh = arr("tedge_manhattan_dist", np.full(E, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    e_hpwl = arr("tedge_net_hpwl", np.full(E, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    e_fanout = arr("tedge_net_fanout", np.full(E, -1, dtype=np.int32)).astype(np.float32).reshape(-1)[:E]
    pmaxx = arr("tedge_path_max_chanx", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    pavgx = arr("tedge_path_avg_chanx", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    pmaxy = arr("tedge_path_max_chany", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    pavgy = arr("tedge_path_avg_chany", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]

    edge_index = np.stack([tedge_src.astype(np.int64), tedge_dst.astype(np.int64)], axis=0)

    delay_n = np.where(tedge_delay >= 0, tedge_delay / cpd, 0.0).astype(np.float32)
    manh_n = np.where(manh >= 0, manh / grid_sum, 0.0).astype(np.float32)
    eh_n = np.where(e_hpwl >= 0, e_hpwl / grid_sum, 0.0).astype(np.float32)
    ef_n = np.where(e_fanout >= 0, np.log1p(np.maximum(e_fanout, 0)), 0.0).astype(np.float32)
    pmx = np.log1p(np.maximum(pmaxx, 0)).astype(np.float32)
    pax = np.log1p(np.maximum(pavgx, 0)).astype(np.float32)
    pmy = np.log1p(np.maximum(pmaxy, 0)).astype(np.float32)
    pay = np.log1p(np.maximum(pavgy, 0)).astype(np.float32)

    etype_oh = _one_hot(tedge_type, 4)
    edge_attr_np = np.concatenate(
        [
            etype_oh,
            delay_n.reshape(-1, 1),
            manh_n.reshape(-1, 1),
            eh_n.reshape(-1, 1),
            ef_n.reshape(-1, 1),
            pmx.reshape(-1, 1),
            pax.reshape(-1, 1),
            pmy.reshape(-1, 1),
            pay.reshape(-1, 1),
        ],
        axis=1,
    )
    assert edge_attr_np.shape[1] == 12

    rt_time = arr("tnode_rt_time", np.full(N, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:N]
    y_valid = arr("tnode_valid_mask", np.zeros(N, dtype=np.int32)).astype(np.bool_).reshape(-1)[:N]
    y_arrival = np.where(y_valid, rt_time / cpd, 0.0).astype(np.float32)

    top_paths = arr("top_path_node_ids")
    if top_paths is not None:
        y_critical = _build_y_critical_from_top_paths(top_paths, N, K=20)
    else:
        crit = arr("tnode_on_critical_path", np.zeros(N, dtype=np.int8))
        y_critical = crit.astype(np.bool_).reshape(-1)[:N].copy()

    data = Data(
        x=torch.from_numpy(x_np),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr_np),
        y_arrival=torch.from_numpy(y_arrival),
        y_valid=torch.from_numpy(y_valid),
        y_critical=torch.from_numpy(y_critical),
        cpd=torch.tensor([cpd], dtype=torch.float32),
    )
    data.node_type = torch.from_numpy(tnode_type.astype(np.int64))
    data.tedge_delay = torch.from_numpy(tedge_delay.astype(np.float32))
    return data
