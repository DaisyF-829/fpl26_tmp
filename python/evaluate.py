"""
加载训练好的时序异构图模型（MPNN 或 gnn 中 GCN/GAT/GraphSAGE/GIN 对照），对单文件或整目录 .npz 评估。

checkpoint 的 `model_class` 决定结构；旧权重无该字段时默认 MPNN。
`--model_type` 可强制指定骨干（须与 state_dict 一致）。

节点下标与 npz 中 tnode 一致（全图加载，无 node_old_indices）。
含图级 graph_head：回归 log(CPD)；报告 log 误差与 CPD 相对误差%。

`--pl_baseline`：不加载模型。有效 PL 仅由 tnode_pl_arrival>0（且有限）判定，不读 PL 掩码键；无该键时回退 tnode_pl_time。
末节点（fanout==0∩y_valid）上 PL 到达 vs tnode_rt_time 的 R²/MAPE，并照常跑 Top-K/Top20 路径与 Coverage。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data_loader import hetero_combined_edges, load_timing_graph
from gnn import HETERO_CONV_MODELS
from metrics import compute_regression_metrics, format_metrics_line
from model import (
    HeteroTimingMPNN,
    HeteroTimingMPNNDelayProp,
    HeteroTimingMPNNDelayPropStage1Only,
    HeteroTimingMPNNDelayPropStage2Only,
    HeteroTimingMPNNMultiHop,
)

# ckpt["model_class"] 字符串 -> 构造函数
_TIMING_CONV_BY_SAVED_NAME: dict[str, type[nn.Module]] = {
    name: cls for cls, name in HETERO_CONV_MODELS.values()
}


def derive_top_k_paths(
    arrival: np.ndarray,
    edge_index: np.ndarray,
    edge_delay: np.ndarray,
    node_type: np.ndarray,
    K: int = 20,
    valid_mask: np.ndarray | None = None,
) -> set[int]:
    """
    从到达时间最大的前 K 个 SINK 出发，沿前驱反向追踪 STA 风格路径：
    每一步选取使 arrival[pred] + delay(pred→cur) 最大的前驱（与 max 到达时间递推一致）。

    valid_mask: 若为 None，则仅要求 arrival 有限；否则仅 True 下标参与（用于 GT 的 tnode_rt_time）。
    """
    N = int(arrival.shape[0])
    arr = np.asarray(arrival, dtype=np.float64).reshape(-1)[:N]
    pred_edges: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    src = edge_index[0]
    dst = edge_index[1]
    ed = np.asarray(edge_delay, dtype=np.float64).reshape(-1)
    for i in range(edge_index.shape[1]):
        s, d = int(src[i]), int(dst[i])
        if 0 <= s < N and 0 <= d < N:
            del_i = float(ed[i]) if i < ed.size else 0.0
            pred_edges[d].append((s, del_i))

    def node_ok(idx: int) -> bool:
        if not (0 <= idx < N):
            return False
        if valid_mask is not None:
            if not bool(valid_mask[idx]):
                return False
        a = arr[idx]
        return bool(np.isfinite(a) and a >= 0.0)

    sink_mask = node_type == 1
    sink_indices = np.where(sink_mask)[0]
    if sink_indices.size == 0:
        return set()

    scores = np.array([arr[i] if node_ok(i) else -np.inf for i in sink_indices], dtype=np.float64)
    order = sink_indices[np.argsort(-scores)][:K]
    nodes: set[int] = set()
    for start in order:
        if not node_ok(int(start)):
            continue
        cur = int(start)
        visited: set[int] = set()
        while 0 <= cur < N and cur not in visited:
            visited.add(cur)
            nodes.add(cur)
            pes = pred_edges[cur]
            if not pes:
                break
            best_p: int | None = None
            best_sc = -np.inf
            for s, del_ in pes:
                if not node_ok(s):
                    continue
                sc = arr[s] + del_
                if sc > best_sc:
                    best_sc = sc
                    best_p = s
            if best_p is None:
                break
            cur = best_p
    return nodes


def derive_top_k_full_paths(
    arrival: np.ndarray,
    edge_index: np.ndarray,
    edge_delay: np.ndarray,
    node_type: np.ndarray,
    K: int = 10,
    valid_mask: np.ndarray | None = None,
) -> list[list[int]]:
    """
    与 derive_top_k_paths 相同的 STA 回溯规则，但返回 Top-K 条“完整路径”（节点序列）。
    每条路径从 SINK 开始沿前驱反向走到无前驱/断链为止。

    注意：路径序列方向为 [sink, ..., source]（反向），方便与回溯一致。
    """
    N = int(arrival.shape[0])
    arr = np.asarray(arrival, dtype=np.float64).reshape(-1)[:N]
    pred_edges: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    src = edge_index[0]
    dst = edge_index[1]
    ed = np.asarray(edge_delay, dtype=np.float64).reshape(-1)
    for i in range(edge_index.shape[1]):
        s, d = int(src[i]), int(dst[i])
        if 0 <= s < N and 0 <= d < N:
            del_i = float(ed[i]) if i < ed.size else 0.0
            pred_edges[d].append((s, del_i))

    def node_ok(idx: int) -> bool:
        if not (0 <= idx < N):
            return False
        if valid_mask is not None and not bool(valid_mask[idx]):
            return False
        a = arr[idx]
        return bool(np.isfinite(a) and a >= 0.0)

    sink_mask = node_type == 1
    sink_indices = np.where(sink_mask)[0]
    if sink_indices.size == 0:
        return []

    scores = np.array([arr[i] if node_ok(int(i)) else -np.inf for i in sink_indices], dtype=np.float64)
    order = sink_indices[np.argsort(-scores)][:K]

    paths: list[list[int]] = []
    for start in order:
        start_i = int(start)
        if not node_ok(start_i):
            continue
        cur = start_i
        visited: set[int] = set()
        path: list[int] = []
        while 0 <= cur < N and cur not in visited:
            visited.add(cur)
            path.append(cur)
            pes = pred_edges[cur]
            if not pes:
                break
            best_p: int | None = None
            best_sc = -np.inf
            for s, del_ in pes:
                if not node_ok(int(s)):
                    continue
                sc = arr[int(s)] + float(del_)
                if sc > best_sc:
                    best_sc = sc
                    best_p = int(s)
            if best_p is None:
                break
            cur = best_p
        if path:
            paths.append(path)
    return paths


def true_critical_nodes_from_npz(
    z,
    N: int,
    edge_index: np.ndarray,
    edge_delay: np.ndarray,
    node_type: np.ndarray,
    K: int,
    *,
    silent: bool = False,
) -> tuple[set[int], np.ndarray, str]:
    """
    优先使用 tnode_on_critical_path；无法读取时用 tnode_rt_time + tedge_delay 做 STA 回溯构造真值。
    注意：此处使用的 tedge_delay 仅用于评估/构造真值，不应作为模型输入特征。
    返回 (节点集合, 长度 N 的 int8 掩码, 来源说明字符串)。
    """
    true_crit = np.zeros(N, dtype=np.int8)
    try:
        raw = z["tnode_on_critical_path"]
        crit_full = np.asarray(raw, dtype=np.int8).reshape(-1)[:N]
        true_crit = crit_full.copy()
        nodes = {int(i) for i in np.where(true_crit > 0)[0]}
        return nodes, true_crit, "npz:tnode_on_critical_path"
    except Exception:
        pass

    rt = z["tnode_rt_time"] if "tnode_rt_time" in z.files else None
    if rt is None:
        if not silent:
            print("无法从 npz 读取 tnode_on_critical_path 或 tnode_rt_time，Coverage/Precision 无真值。")
        return set(), true_crit, "none"

    rt = np.asarray(rt, dtype=np.float64).reshape(-1)[:N]
    if "tnode_valid_mask" in z.files:
        vm = np.asarray(z["tnode_valid_mask"]).reshape(-1)[:N]
        node_vm = vm.astype(bool) if vm.dtype != np.bool_ else vm.copy()
    else:
        node_vm = np.ones(N, dtype=bool)
    valid = node_vm & np.isfinite(rt) & (rt >= 0.0)

    nodes = derive_top_k_paths(
        rt,
        edge_index,
        edge_delay,
        node_type,
        K=K,
        valid_mask=valid,
    )
    for i in nodes:
        if 0 <= i < N:
            true_crit[i] = 1
    if not silent:
        print(
            "未读取 tnode_on_critical_path；已用 STA 回溯（tnode_rt_time + tedge_delay，"
            f"与 evaluate 相同的 top-{K} SINK）构造真值节点集。"
        )
    return nodes, true_crit, "STA:tnode_rt_time+tedge_delay"


def _graph_head_metrics(
    pred_graph: torch.Tensor,
    cpd_tensor: torch.Tensor,
    pl_max_tensor: torch.Tensor,
) -> dict[str, float]:
    """
    graph_head 在 train.py 中回归 log(cpd / pl_max)。
    评估时：预测 CPD = exp(pred_graph) * pl_max；真值 CPD = cpd_tensor。
    注意：真值 CPD / pl_max 仅用于指标，不参与预测值构造。
    """
    pred_log = pred_graph.detach().float().reshape(-1)
    cpd = cpd_tensor.detach().float().reshape(-1)
    pl_max = pl_max_tensor.detach().float().reshape(-1)
    nan = float("nan")
    if pred_log.numel() == 0:
        return {
            "mae_log": nan,
            "rmse_log": nan,
            "pred_log_mean": nan,
            "cpd_true": nan,
            "pred_cpd_mean": nan,
            "rel_cpd_error_pct": nan,
        }
    cpd = cpd.clamp(min=1e-12)
    pl_max = pl_max.clamp(min=1e-12)

    if pred_log.numel() == 1 and cpd.numel() > 1:
        pred_log = pred_log.expand_as(cpd)
    elif cpd.numel() == 1 and pred_log.numel() > 1:
        cpd = cpd.expand_as(pred_log)
    if pl_max.numel() == 1:
        pl_max = pl_max.expand_as(cpd)

    tgt_log = torch.log((cpd / pl_max).clamp(min=1e-12))
    err = (pred_log - tgt_log)
    mae_log = float(err.abs().mean().item())
    rmse_log = float(torch.sqrt(err.pow(2).mean()).item())
    pred_log_mean = float(pred_log.mean().item())

    pred_cpd = torch.exp(pred_log) * pl_max
    pred_cpd_mean = float(pred_cpd.mean().item())
    cpd_mean = float(cpd.mean().item())
    rel = (pred_cpd - cpd).abs() / cpd * 100.0
    rel_cpd_error_pct = float(rel.mean().item())
    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "pred_log_mean": pred_log_mean,
        "cpd_true": cpd_mean,
        "pred_cpd_mean": pred_cpd_mean,
        "rel_cpd_error_pct": rel_cpd_error_pct,
    }


def load_hetero_model(
    model_path: str,
    device: torch.device,
    *,
    model_type: str = "auto",
) -> nn.Module:
    """
    model_type:
      - auto：使用 ckpt['model_class']，缺省 HeteroTimingMPNN
      - mpnn / mpnn_mh / mpnn_delayprop / mpnn_delayprop_s1 / mpnn_delayprop_s2 /
        gcn / gat / sage / gin：强制该骨干（须与 state_dict 匹配）
    """
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    hidden = int(ckpt.get("hidden", 128))
    layers = int(ckpt.get("layers", 6))
    if model_type == "auto":
        name = str(ckpt.get("model_class", "HeteroTimingMPNN"))
    elif model_type == "mpnn":
        name = "HeteroTimingMPNN"
    elif model_type == "mpnn_mh":
        name = "HeteroTimingMPNNMultiHop"
    elif model_type == "mpnn_delayprop":
        name = "HeteroTimingMPNNDelayProp"
    elif model_type == "mpnn_delayprop_s1":
        name = "HeteroTimingMPNNDelayPropStage1Only"
    elif model_type == "mpnn_delayprop_s2":
        name = "HeteroTimingMPNNDelayPropStage2Only"
    elif model_type in HETERO_CONV_MODELS:
        name = HETERO_CONV_MODELS[model_type][1]
    else:
        name = str(ckpt.get("model_class", "HeteroTimingMPNN"))
    conv_cls = _TIMING_CONV_BY_SAVED_NAME.get(name)
    if conv_cls is not None:
        model = conv_cls(hidden_dim=hidden, num_layers=layers).to(device)
    elif name == "HeteroTimingMPNNMultiHop":
        model = HeteroTimingMPNNMultiHop(hidden_dim=hidden, num_layers=layers).to(device)
    elif name == "HeteroTimingMPNNDelayProp":
        model = HeteroTimingMPNNDelayProp(hidden_dim=hidden, num_layers=layers).to(device)
    elif name == "HeteroTimingMPNNDelayPropStage1Only":
        model = HeteroTimingMPNNDelayPropStage1Only(
            hidden_dim=hidden, num_layers=layers
        ).to(device)
    elif name == "HeteroTimingMPNNDelayPropStage2Only":
        model = HeteroTimingMPNNDelayPropStage2Only(
            hidden_dim=hidden, num_layers=layers
        ).to(device)
    else:
        model = HeteroTimingMPNN(hidden_dim=hidden, num_layers=layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"已加载 {name}  hidden={hidden}  layers={layers}  <- {model_path}",
        flush=True,
    )
    return model


def _find_npz_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.npz"))


def _arrival_for_path_tracing(pred_arrival_ns: np.ndarray, *, pl_baseline: bool) -> np.ndarray:
    """
    STA 回溯要求节点到达时间为有限值。PL 基线在无效 PL 处为 nan，会使 SINK/前驱被 node_ok
    全部排除，pred 路径为空（Coverage=0、Precision=nan）。路径类指标将非有限值视为 0.0；
    节点回归仍用原始 pred_arrival_ns（nan + 掩码）。
    """
    a = np.asarray(pred_arrival_ns, dtype=np.float64).reshape(-1).copy()
    if pl_baseline:
        a[~np.isfinite(a)] = 0.0
    return a


def _pl_baseline_arrays_from_npz(npz_path: str, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    与 load_timing_graph 一致：优先 tnode_pl_arrival，无则 tnode_pl_time；另读 tnode_rt_time。
    pl_baseline 专用：不依赖掩码键，仅用 finite(pl_val) & (pl_val > 0) 为有效 PL。
    """
    z = np.load(npz_path, allow_pickle=False)
    try:

        def arr(name: str, default=None):
            if name not in z.files:
                return default
            return z[name]

        pl_arr = arr("tnode_pl_arrival", None)
        if pl_arr is None:
            pl_arr = arr("tnode_pl_time", np.full(N, -1.0, dtype=np.float32))
        pl_time = np.asarray(pl_arr, dtype=np.float64).reshape(-1)[:N]
        pl_ok = np.isfinite(pl_time) & (pl_time > 0.0)
        if pl_ok.any():
            pl_cpd_hat = float(np.max(pl_time[pl_ok]))
        else:
            pl_cpd_hat = float("nan")
        pred_pl_raw = np.full(N, np.nan, dtype=np.float64)
        pred_pl_raw[pl_ok] = pl_time[pl_ok]
        rt_arr = arr("tnode_rt_time", np.full(N, -1.0, dtype=np.float32))
        rt_raw = np.asarray(rt_arr, dtype=np.float64).reshape(-1)[:N]
        return pred_pl_raw, rt_raw, pl_ok, pl_cpd_hat
    finally:
        z.close()


def evaluate_one(
    model: nn.Module | None,
    device: torch.device,
    npz_path: str,
    K: int = 20,
    save_pred: str | None = None,
    *,
    silent: bool = False,
    pl_baseline: bool = False,
) -> dict[str, Any]:
    """单次评估；silent=True 时不打印过程，仅返回指标字典。pl_baseline=True 时不使用 model。"""
    data = load_timing_graph(npz_path)
    data = data.to(device)
    N = int(data["tnode"].x.shape[0])
    pred_norm_np: np.ndarray | None = None
    rt_raw: np.ndarray | None = None
    pl_ok: np.ndarray | None = None
    pl_cpd_hat = float("nan")

    with torch.no_grad():
        if pl_baseline:
            pred_arrival_ns, rt_raw, pl_ok_arr, pl_cpd_hat = _pl_baseline_arrays_from_npz(
                npz_path, N
            )
            pl_ok = pl_ok_arr.astype(bool, copy=False)
            nan = float("nan")
            cpd0 = float(data.cpd.reshape(-1)[0].item())
            m_graph = {
                "mae_log": nan,
                "rmse_log": nan,
                "pred_log_mean": nan,
                "cpd_true": cpd0,
                "pred_cpd_mean": nan,
                "rel_cpd_error_pct": nan,
            }
            pred_graph = torch.tensor([0.0], dtype=torch.float32, device=device)
            plm = getattr(data, "pl_max", data.cpd)
        else:
            if model is None:
                raise ValueError("evaluate_one: 非 pl_baseline 时必须提供 model")
            pred_norm, pred_graph = model(data)
            pred_norm_np = pred_norm.detach().cpu().numpy()
            plm = getattr(data, "pl_max", data.cpd)
            pred_arrival_ns = (pred_norm * plm.reshape(-1)[0]).detach().cpu().numpy()
            m_graph = _graph_head_metrics(pred_graph, data.cpd, plm)

    N = int(pred_arrival_ns.shape[0])
    y_arrival = data["tnode"].y_arrival.detach().cpu().numpy().reshape(-1)[:N]
    y_valid = data["tnode"].y_valid.detach().cpu().numpy().reshape(-1)[:N].astype(bool)
    fanout = data["tnode"].fanout.detach().cpu().numpy().reshape(-1)[:N]
    leaf_mask = (fanout == 0) & y_valid
    n_leaf = int(leaf_mask.sum())
    node_type_np = data["tnode"].node_type.detach().cpu().numpy().reshape(-1)[:N]
    sink_mask = (node_type_np == 1) & y_valid
    n_sink = int(sink_mask.sum())

    m_all: dict[str, float] | None = None
    m_leaf: dict[str, float] | None = None

    if pl_baseline:
        assert rt_raw is not None and pl_ok is not None
        end_mask = leaf_mask & pl_ok & np.isfinite(rt_raw) & (rt_raw >= 0.0)
        n_end = int(end_mask.sum())
        if n_end >= 2:
            m_all = compute_regression_metrics(
                pred_arrival_ns[end_mask].astype(np.float64, copy=False),
                rt_raw[end_mask].astype(np.float64, copy=False),
            )
        if not silent:
            print(
                "\n[PL 基线] 有效PL=tnode_pl_arrival>0（无键则用 tnode_pl_time）；末节点 fanout==0：PL vs tnode_rt_time（未归一化）",
                flush=True,
            )
            print(f"可评估末节点数: {n_end}", flush=True)
            if m_all is not None:
                print(f"R²={m_all['r2']:.4f}  MAPE={m_all['mape']:.2f}%", flush=True)
            else:
                print("（可评估末节点 < 2，不计算 R²/MAPE）", flush=True)
    else:
        if not silent:
            print("\n--- 图级 graph_head（回归 log(CPD/pl_max)）---", flush=True)
            print(
                f"pred_log_mean={m_graph['pred_log_mean']:.6f}  MAE(log)={m_graph['mae_log']:.6f}  "
                f"RMSE(log)={m_graph['rmse_log']:.6f}",
                flush=True,
            )
            print(
                f"真值 CPD={m_graph['cpd_true']:.6g}  预测 CPD=exp(pred_log)*pl_max={m_graph['pred_cpd_mean']:.6g}  "
                f"相对误差={m_graph['rel_cpd_error_pct']:.4f}%",
                flush=True,
            )
            if n_sink > 0:
                plm_f = float(plm.reshape(-1)[0])
                pred_cpd_from_nodes = float(np.max(pred_arrival_ns[sink_mask]))
                true_arrival_ns = (y_arrival * plm_f).astype(np.float64, copy=False)
                true_cpd_from_nodes = float(np.max(true_arrival_ns[sink_mask]))
                denom = max(true_cpd_from_nodes, 1e-12)
                rel_nodes = abs(pred_cpd_from_nodes - true_cpd_from_nodes) / denom * 100.0
                print(
                    f"CPD(由节点复原，max sink arrival): true={true_cpd_from_nodes:.6g}  "
                    f"pred={pred_cpd_from_nodes:.6g}  相对误差={rel_nodes:.4f}%  "
                    f"(sink 可监督点 {n_sink} 个)",
                    flush=True,
                )
            else:
                print("CPD(由节点复原): (无可监督的 SINK 节点，跳过)", flush=True)

    if not pl_baseline:
        if not silent:
            print(
                f"\n--- 全图监督节点（y_valid）上指标（归一化到达时间 rt/pl_max）---",
                flush=True,
            )
        if y_valid.any():
            assert pred_norm_np is not None
            m_all = compute_regression_metrics(pred_norm_np[y_valid], y_arrival[y_valid])
            if not silent:
                print(format_metrics_line(m_all), flush=True)
        else:
            if not silent:
                print("(无 y_valid 节点)", flush=True)

        if not silent:
            print(
                f"--- 末节点（fanout==0 且 y_valid，即全图所有末节点监督点）：共 {n_leaf} 个 ---",
                flush=True,
            )
        if n_leaf >= 2:
            assert pred_norm_np is not None
            m_leaf = compute_regression_metrics(pred_norm_np[leaf_mask], y_arrival[leaf_mask])
            if not silent:
                print(
                    "本图末节点 — τ / Spearman / R² / MAPE:  "
                    f"τ={m_leaf['tau']:.4f}  ρ={m_leaf['spearman_r']:.4f}  "
                    f"R²={m_leaf['r2']:.4f}  MAPE={m_leaf['mape']:.2f}%",
                    flush=True,
                )
                print(
                    f"本图末节点 — MAE / RMSE:  MAE={m_leaf['mae']:.4f}  RMSE={m_leaf['rmse']:.4f}",
                    flush=True,
                )
        else:
            if not silent:
                print("末节点可监督样本不足 2 个，跳过 τ / Spearman / R² / MAPE。", flush=True)

    n_leaf_out = n_leaf
    if pl_baseline:
        n_leaf_out = int(end_mask.sum())

    cei, ctd = hetero_combined_edges(data)
    edge_index = cei.cpu().numpy()
    tedge_delay = ctd.cpu().numpy()
    node_type = data["tnode"].node_type.cpu().numpy()

    path_arrival = _arrival_for_path_tracing(pred_arrival_ns, pl_baseline=pl_baseline)
    pred_nodes = derive_top_k_paths(
        path_arrival,
        edge_index,
        tedge_delay,
        node_type,
        K=K,
    )

    z = np.load(npz_path, allow_pickle=False)
    try:
        true_path_nodes, true_crit, _ = true_critical_nodes_from_npz(
            z, N, edge_index, tedge_delay, node_type, K, silent=silent
        )

        # 额外指标：Top-20 路径重叠检索（按“预测 top20 完整路径的并集集合”做召回）
        top_paths_k = 20
        pred_paths_top20 = derive_top_k_full_paths(
            path_arrival, edge_index, tedge_delay, node_type, K=top_paths_k
        )
        pred_union_nodes_top20: set[int] = set()
        for p in pred_paths_top20:
            pred_union_nodes_top20.update(int(x) for x in p)

        if "tnode_rt_time" in z.files:
            rt = np.asarray(z["tnode_rt_time"], dtype=np.float64).reshape(-1)[:N]
            if "tnode_valid_mask" in z.files:
                vm = np.asarray(z["tnode_valid_mask"]).reshape(-1)[:N]
                node_vm = vm.astype(bool) if vm.dtype != np.bool_ else vm.copy()
            else:
                node_vm = np.ones(N, dtype=bool)
            valid = node_vm & np.isfinite(rt) & (rt >= 0.0)
            true_paths_top20 = derive_top_k_full_paths(
                rt, edge_index, tedge_delay, node_type, K=top_paths_k, valid_mask=valid
            )
        else:
            true_paths_top20 = []

        hit_thr = 0.5
        found_true_paths = 0
        overlap_ratios: list[float] = []
        for tp in true_paths_top20:
            denom = len(tp)
            if denom <= 0:
                continue
            overlap = sum(1 for x in tp if int(x) in pred_union_nodes_top20)
            overlap_ratios.append(overlap / denom)
            if overlap / denom >= hit_thr:
                found_true_paths += 1
    finally:
        z.close()

    inter = pred_nodes & true_path_nodes
    if len(true_path_nodes) > 0:
        coverage = len(inter) / len(true_path_nodes)
    else:
        coverage = float("nan")
    if len(pred_nodes) > 0:
        precision = len(inter) / len(pred_nodes)
    else:
        precision = float("nan")

    if not silent:
        print(f"True critical nodes : {len(true_path_nodes)}")
        print(f"Predicted path nodes: {len(pred_nodes)}")
        print(f"Coverage (Recall)   : {coverage:.3f}" if not np.isnan(coverage) else "Coverage (Recall)   : nan")
        print(f"Precision           : {precision:.3f}" if not np.isnan(precision) else "Precision           : nan")
        if "found_true_paths" in locals():
            denom_t = len(true_paths_top20) if "true_paths_top20" in locals() else 0
            if denom_t > 0:
                print(
                    f"Top20 路径检索(>=50% 命中): found={found_true_paths}/{denom_t}  "
                    f"ratio={found_true_paths / denom_t:.3f}",
                    flush=True,
                )
            else:
                print("Top20 路径检索(>=50% 命中): (无 tnode_rt_time 或无可用真值路径，跳过)", flush=True)

    if save_pred:
        on_pred = np.zeros(N, dtype=np.int8)
        for i in pred_nodes:
            if 0 <= i < N:
                on_pred[i] = 1
        true_is_critical = true_crit.astype(np.int8).reshape(-1)[:N]
        out_path = Path(save_pred)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pg0 = float(pred_graph.detach().float().reshape(-1)[0].item())
        cpd0 = float(data.cpd.reshape(-1)[0].item())
        plm0 = float(plm.reshape(-1)[0].item())
        pred_cpd_out = (
            float(pl_cpd_hat)
            if pl_baseline and math.isfinite(float(pl_cpd_hat))
            else float(np.exp(pg0) * plm0)
        )
        np.savez(
            out_path,
            pred_arrival=pred_arrival_ns.astype(np.float32),
            pred_is_critical=on_pred,
            true_is_critical=true_is_critical,
            pred_graph_log_ratio=np.float32(pg0),
            pl_max=np.float32(plm0),
            pred_cpd=np.float32(pred_cpd_out),
            cpd_true=np.float32(cpd0),
        )
        if not silent:
            print(f"已保存 {out_path}")

    return {
        "path": npz_path,
        "coverage": float(coverage),
        "precision": float(precision),
        "top20_path_found": int(found_true_paths) if "found_true_paths" in locals() else 0,
        "top20_path_total": int(len(true_paths_top20)) if "true_paths_top20" in locals() else 0,
        # Top20 overlap：对“真值 top20 每条路径”的节点覆盖比例做均值（单文件内）
        "top20_overlap_mean": (
            float(np.mean(overlap_ratios)) if "overlap_ratios" in locals() and len(overlap_ratios) > 0 else float("nan")
        ),
        "m_all": m_all,
        "m_leaf": m_leaf,
        "m_graph": m_graph,
        "n_leaf": n_leaf_out,
    }


def evaluate(
    model_path: str,
    npz_path: str,
    K: int = 20,
    save_pred: str | None = None,
    *,
    model_type: str = "auto",
    pl_baseline: bool = False,
) -> dict[str, Any]:
    """兼容旧用法：加载模型并评估单个 npz；pl_baseline 时忽略 model_path。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None if pl_baseline else load_hetero_model(model_path, device, model_type=model_type)
    print(f"文件: {npz_path}", flush=True)
    return evaluate_one(
        model, device, npz_path, K=K, save_pred=save_pred, silent=False, pl_baseline=pl_baseline
    )


def _nanmean(vals: list[float]) -> float:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(a.mean())


def _nanmean_metric(rows: list[dict[str, Any]], key: str, sub: str) -> float:
    vals: list[float] = []
    for r in rows:
        d = r.get(key)
        if d is None:
            continue
        v = d.get(sub)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return _nanmean(vals)


def _n_rows_with_dict(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for r in rows if r.get(key) is not None)


def _nanmean_int(rows: list[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            vals.append(float(v))
    return _nanmean(vals)


def evaluate_directory(
    model_path: str,
    data_dir: str,
    K: int = 20,
    save_pred_dir: str | None = None,
    *,
    quiet: bool = False,
    max_files: int = 0,
    model_type: str = "auto",
    pl_baseline: bool = False,
) -> None:
    root = Path(data_dir)
    paths = _find_npz_files(root)
    if not paths:
        raise SystemExit(f"目录下未找到 .npz: {root}")
    if max_files > 0:
        paths = paths[:max_files]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pl_baseline:
        model = None
        print(
            "PL 基线：用 tnode_pl_arrival（无则 tnode_pl_time）作为节点预测，跳过模型加载。",
            flush=True,
        )
    else:
        model = load_hetero_model(model_path, device, model_type=model_type)
    print(f"目录评估: {root.resolve()}  共 {len(paths)} 个 .npz", flush=True)
    if quiet:
        print("quiet 模式：不逐文件打印，仅输出最终汇总。", flush=True)
    else:
        print("不逐文件打印，仅输出最终汇总（如需逐文件，请自行在代码里打开打印）。", flush=True)
    print("", flush=True)

    rows: list[dict[str, Any]] = []
    for i, p in enumerate(paths, start=1):
        save_one = None
        if save_pred_dir:
            out_d = Path(save_pred_dir)
            save_one = str(out_d / f"{p.stem}_pred.npz")
        try:
            # 用户要求：不逐文件打印；统一 silent=True
            row = evaluate_one(
                model,
                device,
                str(p),
                K=K,
                save_pred=save_one,
                silent=True,
                pl_baseline=pl_baseline,
            )
            rows.append(row)
        except Exception as ex:
            err = str(ex)
            hint = ""
            if "CRC" in err or "BadZipFile" in err or "zipfile.BadZipFile" in err:
                hint = " [npz 为 ZIP：CRC 校验失败，多为文件损坏或未完整拷贝，需重新导出/传输]"
            print(f"[{i}/{len(paths)}] 跳过（错误）: {p} — {ex}{hint}", flush=True)

    if not rows:
        print("无成功评估的文件。", flush=True)
        return

    print(f"\n{'=' * 72}\n汇总（宏平均，按文件） 成功 {len(rows)}/{len(paths)} 个\n{'=' * 72}", flush=True)
    print(
        f"Coverage 均值: {_nanmean([r['coverage'] for r in rows]):.4f}  "
        f"Precision 均值: {_nanmean([r['precision'] for r in rows]):.4f}",
        flush=True,
    )
    if any(("top20_overlap_mean" in r) for r in rows):
        print(
            f"Top20 overlap 均值（对真值 top20 每条路径的节点覆盖比例取均值，再按文件宏平均）: "
            f"{_nanmean([float(r.get('top20_overlap_mean')) for r in rows if isinstance(r.get('top20_overlap_mean'), (int, float))]):.4f}",
            flush=True,
        )
    if any(("top20_path_total" in r) for r in rows):
        totals = [int(r.get("top20_path_total", 0)) for r in rows]
        founds = [int(r.get("top20_path_found", 0)) for r in rows]
        tot_sum = int(np.sum(totals)) if totals else 0
        found_sum = int(np.sum(founds)) if founds else 0
        ratio = (found_sum / tot_sum) if tot_sum > 0 else float("nan")
        print(
            f"Top20 路径检索(>=50% 命中): 总 found={found_sum}/{tot_sum}  ratio={ratio:.4f}  "
            f"(按文件平均 found={_nanmean_int(rows, 'top20_path_found'):.3f}  "
            f"total={_nanmean_int(rows, 'top20_path_total'):.3f})",
            flush=True,
        )
    if (not pl_baseline) and any(r.get("m_graph") for r in rows):
        print(
            "图头(log(CPD)):  "
            f"MAE(log) 均值={_nanmean_metric(rows, 'm_graph', 'mae_log'):.6f}  "
            f"RMSE(log) 均值={_nanmean_metric(rows, 'm_graph', 'rmse_log'):.6f}  "
            f"CPD MAPE(%) 均值={_nanmean_metric(rows, 'm_graph', 'rel_cpd_error_pct'):.4f}",
            flush=True,
        )

    m_all_label = "全图监督节点（每图 y_valid）"
    if pl_baseline:
        m_all_label = (
            "全图监督节点（每图 y_valid）[PL基线=末节点fanout0，tnode_pl_arrival vs rt]"
        )
    for label, key in [(m_all_label, "m_all")]:
        if not any(r.get(key) for r in rows):
            continue
        print(
            f"{label} — 宏平均:  tau={_nanmean_metric(rows, key, 'tau'):.4f}  "
            f"spearman={_nanmean_metric(rows, key, 'spearman_r'):.4f}  "
            f"r2={_nanmean_metric(rows, key, 'r2'):.4f}  "
            f"mape={_nanmean_metric(rows, key, 'mape'):.2f}%  "
            f"mae={_nanmean_metric(rows, key, 'mae'):.4f}  "
            f"rmse={_nanmean_metric(rows, key, 'rmse'):.4f}",
            flush=True,
        )
    def _print_subset_mean(label: str, sub_rows: list[dict[str, Any]]) -> None:
        if not sub_rows:
            print(f"{label}: (0 files)", flush=True)
            return
        cov_m = _nanmean([r["coverage"] for r in sub_rows])
        prec_m = _nanmean([r["precision"] for r in sub_rows])
        msg = f"{label}: n={len(sub_rows)}  cov_mean={cov_m:.4f}  prec_mean={prec_m:.4f}"

        if any(("top20_path_total" in r) for r in sub_rows):
            totals = [int(r.get("top20_path_total", 0)) for r in sub_rows]
            founds = [int(r.get("top20_path_found", 0)) for r in sub_rows]
            tot_sum = int(np.sum(totals)) if totals else 0
            found_sum = int(np.sum(founds)) if founds else 0
            ratio = (found_sum / tot_sum) if tot_sum > 0 else float("nan")
            msg += f"  top20_path={found_sum}/{tot_sum}({ratio:.4f})"

        if any(r.get("m_leaf") for r in sub_rows):
            msg += (
                f"  leaf_tau_mean={_nanmean_metric(sub_rows, 'm_leaf', 'tau'):.4f}"
                f"  leaf_sp_mean={_nanmean_metric(sub_rows, 'm_leaf', 'spearman_r'):.4f}"
                f"  leaf_r2_mean={_nanmean_metric(sub_rows, 'm_leaf', 'r2'):.4f}"
                f"  leaf_mape_mean={_nanmean_metric(sub_rows, 'm_leaf', 'mape'):.2f}%"
            )
        if any(r.get("m_graph") for r in sub_rows):
            msg += f"  cpd_rel%_mean={_nanmean_metric(sub_rows, 'm_graph', 'rel_cpd_error_pct'):.4f}"
        print(msg, flush=True)

    # 额外分组统计：按文件名包含关键字筛选
    group_keys = [
        "bgm",
        "blob_merge",
        "boundtop",
        "ch_intrinsics",
        "diffeq1",
        "diffeq2",
        "mcml",
        "mkDelayWorker32B",
        "mkPktMerge",
        "mkSMAdapter4B",
        "or1200",
        "raygentop",
        "sha",
        "spree",
        "frisc",
        "dsip",
        "s38417",
        "s38584.1",
        "tseng",
    ]
    print(f"\n{'-' * 72}\n分组均值（按 npz 文件名包含关键字）\n{'-' * 72}", flush=True)
    for k in group_keys:
        sub = [r for r in rows if k in Path(str(r.get("path", ""))).name]
        _print_subset_mean(f"[{k}]", sub)
    _print_subset_mean("[routedCLK]", [r for r in rows if "routedCLK" in Path(str(r.get("path", ""))).name])
    _print_subset_mean("[c100]", [r for r in rows if "chan_100" in Path(str(r.get("path", ""))).name])

    n_leaf_ok = _n_rows_with_dict(rows, "m_leaf")
    # 末节点关键数据（总数、参与统计图数、宏平均各指标）
    total_leaf_points = int(np.sum([int(r.get("n_leaf", 0) or 0) for r in rows]))
    if pl_baseline:
        print(
            f"PL 基线：末节点∩有效PL 可评估点数（每图计数之和）: {total_leaf_points}",
            flush=True,
        )
    elif n_leaf_ok > 0:
        print(
            f"末节点（每图内 fanout==0 且 y_valid 的全部点）— 宏平均:  "
            f"总末节点监督点数={total_leaf_points}  "
            f"参与平均的图数={n_leaf_ok}/{len(rows)}（末节点数<2 的图无 τ 等统计，不计入）",
            flush=True,
        )
        print(
            "  τ / Spearman / R² / MAPE 均值:  "
            f"tau={_nanmean_metric(rows, 'm_leaf', 'tau'):.4f}  "
            f"spearman={_nanmean_metric(rows, 'm_leaf', 'spearman_r'):.4f}  "
            f"r2={_nanmean_metric(rows, 'm_leaf', 'r2'):.4f}  "
            f"mape={_nanmean_metric(rows, 'm_leaf', 'mape'):.2f}%",
            flush=True,
        )
        print(
            "  MAE / RMSE 均值:  "
            f"mae={_nanmean_metric(rows, 'm_leaf', 'mae'):.4f}  "
            f"rmse={_nanmean_metric(rows, 'm_leaf', 'rmse'):.4f}",
            flush=True,
        )
    else:
        print(
            f"末节点（fanout==0 且 y_valid）：总末节点监督点数={total_leaf_points}  "
            f"但所有图末节点可监督样本不足 2 个，跳过 τ / Spearman / R² / MAPE。",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="评估时序异构图模型（MPNN / GCN）：单文件或整目录 .npz"
    )
    ap.add_argument("--model", type=str, default="model.pt")
    ap.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=(
            "auto",
            "mpnn",
            "mpnn_mh",
            "mpnn_delayprop",
            "mpnn_delayprop_s1",
            "mpnn_delayprop_s2",
            "gcn",
            "gat",
            "sage",
            "gin",
        ),
        help="骨干：auto=读 checkpoint；mpnn/mpnn_mh/mpnn_delayprop(_s1/_s2)/gcn/gat/sage/gin 强制指定（须与权重一致）",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", type=str, default=None, help="单个 .npz 路径")
    g.add_argument("--npz_dir", type=str, default=None, help="递归评估该目录下所有 .npz")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument(
        "--save_pred",
        type=str,
        default=None,
        help="单文件模式：prediction 输出 .npz 路径",
    )
    ap.add_argument(
        "--save_pred_dir",
        type=str,
        default=None,
        help="目录模式：将每个图的预测保存为该目录下 {原名}_pred.npz",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="目录模式：每个文件只打印一行摘要；单文件模式勿用",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="目录模式最多处理多少个文件（0 表示不限制）",
    )
    ap.add_argument(
        "--pl_baseline",
        action="store_true",
        help="不加载 --model；仅末节点 fanout==0：tnode_pl_arrival vs tnode_rt_time 的 R² 与 MAPE（无 arrival 键则 pl_time）",
    )
    args = ap.parse_args()

    if args.npz:
        evaluate(
            args.model,
            args.npz,
            K=args.K,
            save_pred=args.save_pred,
            model_type=args.model_type,
            pl_baseline=args.pl_baseline,
        )
    else:
        evaluate_directory(
            args.model,
            args.npz_dir,
            K=args.K,
            save_pred_dir=args.save_pred_dir,
            quiet=args.quiet,
            max_files=args.max_files,
            model_type=args.model_type,
            pl_baseline=args.pl_baseline,
        )


if __name__ == "__main__":
    main()
