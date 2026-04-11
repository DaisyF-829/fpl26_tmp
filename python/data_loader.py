"""
将 C++ 导出的 timing_graph.npz 转为 torch_geometric.data.HeteroData。

- 保留 npz 中全部 N 个 tnode，节点下标与 npz 一致（便于 evaluate 对齐）。
- 4 种 tedge_type 拆成 4 类异构边 (tnode, e{k}, tnode)，边特征 8 维（无 etype one-hot）。
- y_valid = tnode_valid_mask & (tnode_rt_time 有限且 >= 0)；y_arrival = rt_time/pl_max（无效为 0；pl_max 定义见下）。
- pl_max：掩码有效且 tnode_pl_arrival（无则 tnode_pl_time）有限、>=0 时的最大值（图头监督 log(cpd/pl_max)；无此类点时取 cpd）。
- 路径分析可用 hetero_combined_edges(data) 将 e0→e3 边与 delay 拼成一张同构图（不参与 batch 默认 collate）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    idx = indices.astype(np.int64).clip(0, num_classes - 1)
    oh = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(idx.shape[0]), idx] = 1.0
    return oh


NUM_EDGE_TYPES = 4
NODE_KEY = "tnode"


def edge_type_key(k: int) -> tuple[str, str, str]:
    return (NODE_KEY, f"e{k}", NODE_KEY)


def hetero_combined_edges(data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
    """
    按 e0→e3 顺序拼接 edge_index 与 tedge_delay（评估/回溯用，不作为模型输入特征），与 data 同 device。
    """
    device = data[NODE_KEY].x.device
    srcs: list[torch.Tensor] = []
    dsts: list[torch.Tensor] = []
    dels: list[torch.Tensor] = []
    for k in range(NUM_EDGE_TYPES):
        rel = edge_type_key(k)
        ei = data[rel].edge_index
        if ei.numel() == 0:
            continue
        srcs.append(ei[0])
        dsts.append(ei[1])
        td = getattr(data[rel], "tedge_delay", None)
        if td is None:
            td = torch.zeros(ei.size(1), dtype=torch.float32, device=device)
        dels.append(td.to(device))
    if not srcs:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return torch.stack([z, z]), torch.zeros(0, dtype=torch.float32, device=device)
    return (
        torch.stack([torch.cat(srcs), torch.cat(dsts)]),
        torch.cat(dels) if dels else torch.zeros(torch.cat(srcs).numel(), dtype=torch.float32, device=device),
    )


def load_timing_graph(npz_path: str) -> HeteroData:
    z = np.load(npz_path, allow_pickle=False)

    def arr(name: str, default=None):
        if name not in z.files:
            return default
        return z[name]

    tnode_type = arr("tnode_type")
    if tnode_type is None:
        z.close()
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

    pl_arr = arr("tnode_pl_arrival", None)
    if pl_arr is None:
        pl_arr = arr("tnode_pl_time", np.full(N, -1.0, dtype=np.float32))
    pl_time = np.asarray(pl_arr, dtype=np.float32).reshape(-1)[:N]
    pl_raw = arr("tnode_pl_arrival_mask", arr("tnode_pl_valid", None))
    if pl_raw is None:
        pl_valid = np.zeros(N, dtype=np.float32)
    else:
        pl_valid = np.asarray(pl_raw).astype(np.float32).reshape(-1)[:N]

    # 有效 PL：掩码为真且 pl_time 有限、非负（避免 nan/-1 等污染 pl_max 与特征）
    pl_ok = (pl_valid > 0) & np.isfinite(pl_time) & (pl_time >= 0.0)
    # 有效 PL 到达时间的最大值；图头监督 log(cpd/pl_max)；无有效 PL 时退化为 cpd
    if pl_ok.any():
        pl_max = float(np.max(pl_time[pl_ok]))
    else:
        pl_max = float(cpd)
    if pl_max <= 0 or not np.isfinite(pl_max):
        pl_max = float(cpd)

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

    pl_feat = np.where(pl_ok, pl_time / cpd, 0.0).astype(np.float32)

    x_coord = np.where(tnode_x >= 0, tnode_x / float(gw_denom), -0.05).astype(np.float32)
    y_coord = np.where(tnode_y >= 0, tnode_y / float(gh_denom), -0.05).astype(np.float32)

    hpwl_n = np.where(net_hpwl_n >= 0, net_hpwl_n / grid_sum, 0.0).astype(np.float32)
    nf_n = np.where(net_fanout_n >= 0, np.log1p(np.maximum(net_fanout_n, 0)), 0.0).astype(np.float32)

    tnode_type_i = np.asarray(tnode_type).astype(np.int64).reshape(-1)[:N]
    type_oh = _one_hot(tnode_type_i, 5)
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

    if "tnode_valid_mask" in z.files:
        vm = np.asarray(z["tnode_valid_mask"]).reshape(-1)[:N]
        if vm.dtype == np.bool_:
            node_vm = vm.copy()
        else:
            node_vm = vm.astype(np.float32) != 0
    else:
        node_vm = np.ones(N, dtype=bool)

    rt_time = arr("tnode_rt_time", np.full(N, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:N]
    y_valid_np = node_vm & np.isfinite(rt_time) & (rt_time >= 0.0)
    y_arrival = np.where(y_valid_np, rt_time / float(pl_max), 0.0).astype(np.float32)

    tedge_src = arr("tedge_src")
    tedge_dst = arr("tedge_dst")
    if tedge_src is None or tedge_dst is None:
        z.close()
        raise KeyError(f"{npz_path} 缺少 tedge_src / tedge_dst")

    src_full = np.asarray(tedge_src, dtype=np.int64).reshape(-1)
    dst_full = np.asarray(tedge_dst, dtype=np.int64).reshape(-1)
    E = min(src_full.size, dst_full.size)
    src_full = src_full[:E]
    dst_full = dst_full[:E]

    tedge_type = arr("tedge_type", np.zeros(E, dtype=np.uint64))
    tedge_type = np.asarray(tedge_type, dtype=np.int64).reshape(-1)[:E]
    tedge_type = np.clip(tedge_type, 0, NUM_EDGE_TYPES - 1)

    # 注意：tedge_delay 仅用于评估阶段的路径回溯/构造真值，不进入 edge_attr 以避免训练数据泄露
    tedge_delay = arr("tedge_delay", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    manh = arr("tedge_manhattan_dist", np.full(E, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    e_hpwl = arr("tedge_net_hpwl", np.full(E, -1.0, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    e_fanout = arr("tedge_net_fanout", np.full(E, -1, dtype=np.int32)).astype(np.float32).reshape(-1)[:E]
    pmaxx = arr("tedge_path_max_chanx", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    pavgx = arr("tedge_path_avg_chanx", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    pmaxy = arr("tedge_path_max_chany", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]
    pavgy = arr("tedge_path_avg_chany", np.zeros(E, dtype=np.float32)).astype(np.float32).reshape(-1)[:E]

    manh_n = np.where(manh >= 0, manh / grid_sum, 0.0).astype(np.float32)
    eh_n = np.where(e_hpwl >= 0, e_hpwl / grid_sum, 0.0).astype(np.float32)
    ef_n = np.where(e_fanout >= 0, np.log1p(np.maximum(e_fanout, 0)), 0.0).astype(np.float32)
    pmx = np.log1p(np.maximum(pmaxx, 0)).astype(np.float32)
    pax = np.log1p(np.maximum(pavgx, 0)).astype(np.float32)
    pmy = np.log1p(np.maximum(pmaxy, 0)).astype(np.float32)
    pay = np.log1p(np.maximum(pavgy, 0)).astype(np.float32)

    # 删除 tedge_delay 以避免数据泄露：边特征不包含任何真实延迟/时序信息
    edge_attr_7 = np.stack([manh_n, eh_n, ef_n, pmx, pax, pmy, pay], axis=1).astype(np.float32)
    assert edge_attr_7.shape[1] == 7

    data = HeteroData()
    data[NODE_KEY].x = torch.from_numpy(x_np)
    data[NODE_KEY].y_arrival = torch.from_numpy(y_arrival)
    data[NODE_KEY].y_valid = torch.from_numpy(y_valid_np)
    data[NODE_KEY].y_critical = torch.zeros(N, dtype=torch.bool)
    data[NODE_KEY].node_type = torch.from_numpy(tnode_type_i)
    data[NODE_KEY].fanout = torch.from_numpy(
        np.asarray(fanout, dtype=np.int32).reshape(-1)[:N].copy()
    )

    for k in range(NUM_EDGE_TYPES):
        rel = edge_type_key(k)
        m = tedge_type == k
        if not np.any(m):
            data[rel].edge_index = torch.empty((2, 0), dtype=torch.long)
            data[rel].edge_attr = torch.empty((0, 7), dtype=torch.float32)
            data[rel].tedge_delay = torch.empty((0,), dtype=torch.float32)
            continue
        sk = src_full[m]
        dk = dst_full[m]
        data[rel].edge_index = torch.from_numpy(np.stack([sk, dk])).long()
        data[rel].edge_attr = torch.from_numpy(edge_attr_7[m].copy())
        data[rel].tedge_delay = torch.from_numpy(tedge_delay[m].astype(np.float32).copy())

    data.cpd = torch.tensor([cpd], dtype=torch.float32)
    data.pl_max = torch.tensor([pl_max], dtype=torch.float32)

    z.close()
    return data
