"""
将 C++ 导出的 timing_graph.npz 转为 torch_geometric.data.Data。

- 按 tnode_valid_mask 为 False 的节点从图中删除，并重映射边。
- 回归标签为 tnode_rt_time（归一化到 cpd）；y_valid 由 rt_time（有限且 >=0）推导，不使用 tnode_rt_valid。
- 不读取 tnode_on_critical_path；y_critical 置为全 False（占位，不参与当前特征/损失）。
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


def node_keep_indices(z, N: int) -> np.ndarray:
    """
    与 load_timing_graph 一致：返回保留的原始节点下标（升序）。
    无 tnode_valid_mask 时保留全部节点。
    """
    if "tnode_valid_mask" not in z.files:
        return np.arange(N, dtype=np.int64)
    m = np.asarray(z["tnode_valid_mask"]).reshape(-1)[:N]
    if m.dtype == np.bool_:
        keep = m.copy()
    else:
        keep = m.astype(np.float32) != 0
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError("tnode_valid_mask 全为 False，过滤后无节点")
    return idx.astype(np.int64)


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
    keep_idx = node_keep_indices(z, N)
    Nk = int(keep_idx.size)

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

    def take1d(a: np.ndarray | None, fill, dtype) -> np.ndarray:
        if a is None:
            x = np.full(N, fill, dtype=dtype)
        else:
            x = np.asarray(a).astype(dtype).reshape(-1)[:N]
        return x[keep_idx]

    pl_time = take1d(arr("tnode_pl_time"), -1.0, np.float32)
    pl_raw = arr("tnode_pl_arrival_mask", arr("tnode_pl_valid", None))
    if pl_raw is None:
        pl_valid = np.zeros(Nk, dtype=np.float32)
    else:
        pl_valid = np.asarray(pl_raw).astype(np.float32).reshape(-1)[:N][keep_idx]

    tnode_x = take1d(arr("tnode_x"), 0, np.float32)
    tnode_y = take1d(arr("tnode_y"), 0, np.float32)
    fanin = take1d(arr("tnode_fanin"), 0, np.float32)
    fanout = take1d(arr("tnode_fanout"), 0, np.float32)
    topo = take1d(arr("tnode_topo_level"), -1.0, np.float32)
    net_hpwl_n = take1d(arr("tnode_net_hpwl"), -1.0, np.float32)
    net_fanout_n = take1d(arr("tnode_net_fanout"), -1.0, np.float32)
    tnode_type_k = np.asarray(tnode_type).reshape(-1)[:N][keep_idx].astype(np.int64)

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

    type_oh = _one_hot(tnode_type_k, 5)
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
        z.close()
        raise KeyError(f"{npz_path} 缺少 tedge_src / tedge_dst")

    src_full = np.asarray(tedge_src, dtype=np.int64).reshape(-1)
    dst_full = np.asarray(tedge_dst, dtype=np.int64).reshape(-1)
    E_full = int(src_full.shape[0])
    src_full = src_full[:E_full]
    dst_full = dst_full[:E_full]

    old_to_new = np.full(N, -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(Nk, dtype=np.int64)
    ns = old_to_new[src_full]
    nd = old_to_new[dst_full]
    edge_ok = (ns >= 0) & (nd >= 0)
    new_src = ns[edge_ok]
    new_dst = nd[edge_ok]
    edge_index = np.stack([new_src, new_dst], axis=0)

    tedge_type = arr("tedge_type", np.zeros(E_full, dtype=np.uint64))
    tedge_delay = arr("tedge_delay", np.zeros(E_full, dtype=np.float32))
    manh = arr("tedge_manhattan_dist", np.full(E_full, -1.0, dtype=np.float32))
    e_hpwl = arr("tedge_net_hpwl", np.full(E_full, -1.0, dtype=np.float32))
    e_fanout = arr("tedge_net_fanout", np.full(E_full, -1, dtype=np.int32))
    pmaxx = arr("tedge_path_max_chanx", np.zeros(E_full, dtype=np.float32))
    pavgx = arr("tedge_path_avg_chanx", np.zeros(E_full, dtype=np.float32))
    pmaxy = arr("tedge_path_max_chany", np.zeros(E_full, dtype=np.float32))
    pavgy = arr("tedge_path_avg_chany", np.zeros(E_full, dtype=np.float32))

    def take_edge(a: np.ndarray | None, fill, dtype) -> np.ndarray:
        if a is None:
            x = np.full(E_full, fill, dtype=dtype)
        else:
            x = np.asarray(a).astype(dtype).reshape(-1)[:E_full]
        return x[edge_ok]

    tedge_type = take_edge(tedge_type, 0, np.int64)
    tedge_delay = take_edge(tedge_delay, 0.0, np.float32)
    manh = take_edge(manh, -1.0, np.float32)
    e_hpwl = take_edge(e_hpwl, -1.0, np.float32)
    e_fanout = take_edge(e_fanout, -1.0, np.float32)
    pmaxx = take_edge(pmaxx, 0.0, np.float32)
    pavgx = take_edge(pavgx, 0.0, np.float32)
    pmaxy = take_edge(pmaxy, 0.0, np.float32)
    pavgy = take_edge(pavgy, 0.0, np.float32)

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

    rt_time = arr("tnode_rt_time", np.full(N, -1.0, dtype=np.float32))
    rt_time = np.asarray(rt_time, dtype=np.float32).reshape(-1)[:N][keep_idx]
    y_valid_np = np.isfinite(rt_time) & (rt_time >= 0.0)
    y_arrival = np.where(y_valid_np, rt_time / cpd, 0.0).astype(np.float32)

    y_critical = np.zeros(Nk, dtype=np.bool_)

    data = Data(
        x=torch.from_numpy(x_np),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr_np),
        y_arrival=torch.from_numpy(y_arrival),
        y_valid=torch.from_numpy(y_valid_np),
        y_critical=torch.from_numpy(y_critical),
        cpd=torch.tensor([cpd], dtype=torch.float32),
    )
    data.node_type = torch.from_numpy(tnode_type_k)
    data.tedge_delay = torch.from_numpy(tedge_delay.astype(np.float32))
    data.node_old_indices = torch.from_numpy(keep_idx.copy())

    z.close()
    return data
