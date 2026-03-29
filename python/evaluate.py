"""
加载训练好的 HeteroTimingMPNN，对单个 timing_graph.npz 推理并评估 top-K 路径覆盖率。
节点下标与 npz 中 tnode 一致（全图加载，无 node_old_indices）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loader import hetero_combined_edges, load_timing_graph
from metrics import compute_regression_metrics, format_metrics_line
from model import HeteroTimingMPNN


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


def true_critical_nodes_from_npz(
    z,
    N: int,
    edge_index: np.ndarray,
    edge_delay: np.ndarray,
    node_type: np.ndarray,
    K: int,
) -> tuple[set[int], np.ndarray, str]:
    """
    优先使用 tnode_on_critical_path；无法读取时用 tnode_rt_time + tedge_delay 做 STA 回溯。
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
    print(
        "未读取 tnode_on_critical_path；已用 STA 回溯（tnode_rt_time + tedge_delay，"
        f"与 evaluate 相同的 top-{K} SINK）构造真值节点集。"
    )
    return nodes, true_crit, "STA:tnode_rt_time+tedge_delay"


def evaluate(
    model_path: str,
    npz_path: str,
    K: int = 20,
    save_pred: str | None = None,
) -> tuple[float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    hidden = int(ckpt.get("hidden", 128))
    layers = int(ckpt.get("layers", 6))
    model = HeteroTimingMPNN(hidden_dim=hidden, num_layers=layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = load_timing_graph(npz_path)
    data = data.to(device)
    with torch.no_grad():
        pred_norm, _ = model(data)
    pred_norm_np = pred_norm.detach().cpu().numpy()
    pred_arrival_ns = (pred_norm * data.cpd[0]).detach().cpu().numpy()

    N = int(pred_arrival_ns.shape[0])
    y_arrival = data["tnode"].y_arrival.detach().cpu().numpy().reshape(-1)[:N]
    y_valid = data["tnode"].y_valid.detach().cpu().numpy().reshape(-1)[:N].astype(bool)
    fanout = data["tnode"].fanout.detach().cpu().numpy().reshape(-1)[:N]
    leaf_mask = (fanout == 0) & y_valid
    n_leaf = int(leaf_mask.sum())
    print(f"\n--- 全图监督节点（y_valid）上指标（归一化到达时间）---", flush=True)
    if y_valid.any():
        m_all = compute_regression_metrics(pred_norm_np[y_valid], y_arrival[y_valid])
        print(format_metrics_line(m_all), flush=True)
    else:
        print("(无 y_valid 节点)", flush=True)

    print(f"--- 末节点（fanout==0 且 y_valid）：共 {n_leaf} 个点 ---", flush=True)
    if n_leaf >= 2:
        m_leaf = compute_regression_metrics(pred_norm_np[leaf_mask], y_arrival[leaf_mask])
        print(
            f"排序相关: Kendall τ={m_leaf['tau']:.4f}  Spearman ρ={m_leaf['spearman_r']:.4f}  |  "
            f"R²={m_leaf['r2']:.4f}  MAPE={m_leaf['mape']:.2f}%  "
            f"MAE={m_leaf['mae']:.4f}  RMSE={m_leaf['rmse']:.4f}",
            flush=True,
        )
    else:
        print("末节点可监督样本不足 2 个，跳过 τ / Spearman / R² / MAPE。", flush=True)

    cei, ctd = hetero_combined_edges(data)
    edge_index = cei.cpu().numpy()
    tedge_delay = ctd.cpu().numpy()
    node_type = data["tnode"].node_type.cpu().numpy()

    pred_nodes = derive_top_k_paths(
        pred_arrival_ns,
        edge_index,
        tedge_delay,
        node_type,
        K=K,
    )

    z = np.load(npz_path, allow_pickle=False)
    try:
        true_path_nodes, true_crit, _ = true_critical_nodes_from_npz(
            z, N, edge_index, tedge_delay, node_type, K
        )
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

    print(f"True critical nodes : {len(true_path_nodes)}")
    print(f"Predicted path nodes: {len(pred_nodes)}")
    print(f"Coverage (Recall)   : {coverage:.3f}" if not np.isnan(coverage) else "Coverage (Recall)   : nan")
    print(f"Precision           : {precision:.3f}" if not np.isnan(precision) else "Precision           : nan")

    if save_pred:
        on_pred = np.zeros(N, dtype=np.int8)
        for i in pred_nodes:
            if 0 <= i < N:
                on_pred[i] = 1
        true_is_critical = true_crit.astype(np.int8).reshape(-1)[:N]
        out_path = Path(save_pred)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            pred_arrival=pred_arrival_ns.astype(np.float32),
            pred_is_critical=on_pred,
            true_is_critical=true_is_critical,
        )
        print(f"已保存 {out_path}")

    return (coverage, precision)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="model.pt")
    ap.add_argument("--npz", type=str, required=True, help="单个 timing_graph.npz 路径")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--save_pred", type=str, default=None, help="可选：prediction.npz 输出路径")
    args = ap.parse_args()
    evaluate(args.model, args.npz, K=args.K, save_pred=args.save_pred)


if __name__ == "__main__":
    main()
