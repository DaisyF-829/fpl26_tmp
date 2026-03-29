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
from model import HeteroTimingMPNN


def derive_top_k_paths(
    pred_arrival: np.ndarray,
    edge_index: np.ndarray,
    edge_delay: np.ndarray,
    node_type: np.ndarray,
    K: int = 20,
) -> set[int]:
    """
    从 pred_arrival 最大的前 K 个 SINK 出发，沿前驱反向贪心（前驱里 pred 最大）追踪路径，
    返回所有落在这些路径上的节点索引集合。edge_delay 预留接口，当前未参与决策。
    """
    _ = edge_delay
    N = int(pred_arrival.shape[0])
    predecessors: list[list[int]] = [[] for _ in range(N)]
    src = edge_index[0]
    dst = edge_index[1]
    for i in range(edge_index.shape[1]):
        s, d = int(src[i]), int(dst[i])
        if 0 <= s < N and 0 <= d < N:
            predecessors[d].append(s)

    sink_mask = node_type == 1
    sink_indices = np.where(sink_mask)[0]
    if sink_indices.size == 0:
        return set()

    order = sink_indices[np.argsort(-pred_arrival[sink_indices])][:K]
    nodes: set[int] = set()
    for start in order:
        cur = int(start)
        visited: set[int] = set()
        while 0 <= cur < N and cur not in visited:
            visited.add(cur)
            nodes.add(cur)
            ps = predecessors[cur]
            if not ps:
                break
            cur = max(ps, key=lambda u: pred_arrival[u])
    return nodes


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
        pred_norm = model(data)
    pred_arrival_ns = (pred_norm * data.cpd[0]).detach().cpu().numpy()

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

    N = int(pred_arrival_ns.shape[0])
    z = np.load(npz_path, allow_pickle=False)
    try:
        raw = z["tnode_on_critical_path"]
        crit_full = np.asarray(raw, dtype=np.int8).reshape(-1)[:N]
        true_crit = crit_full
        true_path_nodes = set(int(i) for i in np.where(true_crit > 0)[0])
    except Exception:
        true_crit = np.zeros(N, dtype=np.int8)
        true_path_nodes = set()
        print("未使用 tnode_on_critical_path（无法读取或未启用），Coverage/Precision 参考意义有限。")
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
