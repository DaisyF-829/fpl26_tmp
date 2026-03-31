"""
训练 HeteroTimingMPNN：HeteroData（4 类边），MSE（tnode.y_valid 掩码），按验证集 Kendall τ 保存最优模型。
默认从 data_dir 按 npz 文件三路划分：先 test，再 val，其余 train（无重叠）。
--val_dir：验证集改来自单独目录（忽略 val_frac）；--test_dir：测试集改来自单独目录（忽略 test_frac）。
早停：至少 min_epochs（默认 100）轮；之后验证 tau 连续 patience（默认 20）轮未提升则停止。
训练结束后加载最优 checkpoint 对测试集跑一次 eval。
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from data_loader import load_timing_graph
from gnn import HETERO_CONV_MODELS
from metrics import compute_regression_metrics, format_metrics_line
from model import HeteroTimingMPNN, HeteroTimingMPNNMultiHop


def _find_npz_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    if not root.is_dir():
        return paths
    for p in root.rglob("*.npz"):
        paths.append(p)
    return sorted(paths)


def _split_train_val_only(
    all_paths: list[Path], *, val_frac: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """仅从同一目录划分 train / val（用于已指定独立 test_dir 时）。"""
    n = len(all_paths)
    if n == 0:
        return [], []
    if n == 1:
        return [all_paths[0]], [all_paths[0]]
    vf = min(max(val_frac, 0.0), 0.99)
    n_val = max(1, int(round(vf * n)))
    if n_val >= n:
        n_val = n - 1
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    val_set = set(order[:n_val])
    train_paths = [all_paths[i] for i in order if i not in val_set]
    val_paths_list = [all_paths[i] for i in order if i in val_set]
    if not train_paths:
        train_paths, val_paths_list = val_paths_list[:-1], val_paths_list[-1:]
    return train_paths, val_paths_list


def _split_npz_paths_by_file(
    all_paths: list[Path], *, val_frac: float, test_frac: float, seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """先划 test，再从剩余顺序中划 val，其余 train。同一 npz 只出现在一个集合。"""
    n = len(all_paths)
    if n == 0:
        return [], [], []
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)

    tf = min(max(test_frac, 0.0), 0.99)
    vf = min(max(val_frac, 0.0), 0.99)
    n_test = max(1, int(round(tf * n)))
    n_val = max(1, int(round(vf * n)))
    if n_test + n_val >= n:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
    while n_test + n_val >= n and n > 2:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break

    if n == 1:
        p = all_paths[order[0]]
        return [p], [p], [p]

    test_set = set(order[:n_test])
    val_set = set(order[n_test : n_test + n_val])
    train_paths = [all_paths[i] for i in order if i not in test_set and i not in val_set]
    val_paths_list = [all_paths[i] for i in order if i in val_set]
    test_paths = [all_paths[i] for i in order if i in test_set]

    if not train_paths and val_paths_list:
        train_paths.append(val_paths_list.pop(0))
    if not train_paths and test_paths:
        train_paths.append(test_paths.pop(0))
    if not val_paths_list and train_paths:
        val_paths_list.append(train_paths.pop(-1))

    return train_paths, val_paths_list, test_paths


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    graph_loss_weight: float,
) -> float:
    model.train()
    total_loss_sum = 0.0
    n_batches = 0
    crit = nn.MSELoss(reduction="sum")
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred, pred_graph = model(batch)
        mask = batch["tnode"].y_valid
        # 图级 graph_head：直接回归 CPD（用 log 空间更稳定），评估时不使用真值 CPD 参与预测计算
        cpd = batch.cpd.detach().float().reshape(-1)
        cpd = cpd.clamp(min=1e-12)
        tgt = torch.log(cpd)
        pred_g = pred_graph.float().reshape(-1)
        # pred_graph 允许为标量或 [B]；对齐到 batch 大小
        if pred_g.numel() == 1 and tgt.numel() > 1:
            pred_g = pred_g.expand_as(tgt)
        elif tgt.numel() == 1 and pred_g.numel() > 1:
            tgt = tgt.expand_as(pred_g)
        loss_graph = F.mse_loss(pred_g, tgt)
        if mask.any():
            loss_node = crit(pred[mask], batch["tnode"].y_arrival[mask])
            n = int(mask.sum().item())
            loss_node = loss_node / n
        else:
            loss_node = pred.new_tensor(0.0)
        loss = loss_node + graph_loss_weight * loss_graph
        loss.backward()
        optimizer.step()
        total_loss_sum += float(loss.item())
        n_batches += 1
    if n_batches == 0:
        return 0.0
    return total_loss_sum / n_batches


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    n_mae = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch)
            mask = batch["tnode"].y_valid
            if not mask.any():
                continue
            p = pred[mask].detach().cpu().numpy()
            t = batch["tnode"].y_arrival[mask].detach().cpu().numpy()
            preds.append(p)
            targets.append(t)
            n_mae += p.size

    if n_mae == 0:
        nan = float("nan")
        return {
            "mae": nan,
            "rmse": nan,
            "tau": nan,
            "spearman_r": nan,
            "mape": nan,
            "r2": nan,
        }

    p_all = np.concatenate(preds)
    t_all = np.concatenate(targets)
    return compute_regression_metrics(p_all, t_all)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="递归收集其下所有 .npz")
    ap.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="若指定：训练仅用 data_dir 下 npz，验证仅用 val_dir 下 npz（忽略 --val_frac）",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="未指定 val_dir 时：验证集占 data_dir 中 npz 比例（与 test 划分互斥于剩余部分，见三路划分）",
    )
    ap.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="从 data_dir 中划出的测试集比例（默认 0.1）；指定 --test_dir 时忽略",
    )
    ap.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="若指定：测试集来自该目录，不从 data_dir 中按 test_frac 划分",
    )
    ap.add_argument("--epochs", type=int, default=100, help="最大训练轮数（上限）；早停仍受此限制")
    ap.add_argument(
        "--min_epochs",
        type=int,
        default=100,
        help="至少训练多少个 epoch 后才允许因早停结束",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=20,
        help="验证集 tau 连续多少个 epoch 未刷新最优则早停（仅在 epoch>=min_epochs 时生效）",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--model_type",
        type=str,
        default="mpnn",
        choices=("mpnn", "mpnn_mh", "gcn", "gat", "sage", "gin"),
        help="骨干：mpnn=异构 MPNN；gcn/gat/sage/gin=gnn 中平凡异构卷积对照（HeteroConv+各 conv）",
    )
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument(
        "--graph_loss_weight",
        type=float,
        default=1.0,
        help="全图 max-pool 预测归一化 CPD（目标恒为 1，与 y_arrival 量纲一致）的 MSE 相对节点损失的权重（总 loss = 节点 + weight * 图）",
    )
    ap.add_argument("--save", type=str, default="model.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_dir)
    all_paths = _find_npz_files(data_root)
    if not all_paths:
        raise SystemExit(f"在 {data_root} 下未找到 .npz")

    test_paths: list[Path]

    if args.val_dir:
        val_root = Path(args.val_dir)
        val_paths_list = _find_npz_files(val_root)
        if not val_paths_list:
            raise SystemExit(f"在 {val_root} 下未找到 .npz")
        train_paths = all_paths
        if not train_paths:
            raise SystemExit(f"在 {data_root} 下未找到 .npz")
        if args.test_dir:
            test_root = Path(args.test_dir)
            test_paths = _find_npz_files(test_root)
            if not test_paths:
                raise SystemExit(f"在 {test_root} 下未找到 .npz")
            print(
                f"数据划分：目录模式 — 训练 {len(train_paths)}（{data_root}），"
                f"验证 {len(val_paths_list)}（{val_root}），测试 {len(test_paths)}（{test_root}）。",
                flush=True,
            )
        else:
            test_paths = []
            print(
                f"数据划分：目录模式 — 训练 {len(train_paths)} 个 npz（{data_root}），"
                f"验证 {len(val_paths_list)} 个 npz（{val_root}）；未指定 --test_dir，无独立测试集。",
                flush=True,
            )
    elif args.test_dir:
        test_root = Path(args.test_dir)
        test_paths = _find_npz_files(test_root)
        if not test_paths:
            raise SystemExit(f"在 {test_root} 下未找到 .npz")
        train_paths, val_paths_list = _split_train_val_only(
            all_paths, val_frac=args.val_frac, seed=args.seed
        )
        print(
            f"数据划分：测试目录 — 训练 {len(train_paths)}，验证 {len(val_paths_list)}（均来自 {data_root}，"
            f"val_frac={args.val_frac}），测试 {len(test_paths)}（{test_root}）。",
            flush=True,
        )
    else:
        train_paths, val_paths_list, test_paths = _split_npz_paths_by_file(
            all_paths,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
        print(
            f"数据划分：三路 — 共 {len(all_paths)} 个 npz，训练 {len(train_paths)}，"
            f"验证 {len(val_paths_list)}，测试 {len(test_paths)} "
            f"（val_frac={args.val_frac}, test_frac={args.test_frac}, seed={args.seed}）。",
            flush=True,
        )
        if len(all_paths) == 1:
            print("警告：仅 1 个 npz，训练/验证/测试为同一文件。", flush=True)
        _show = min(3, len(val_paths_list))
        if _show:
            print("验证集 npz 示例（最多 3 个）:", flush=True)
            for p in val_paths_list[:_show]:
                print(f"  {p}", flush=True)
        _show_t = min(3, len(test_paths))
        if _show_t:
            print("测试集 npz 示例（最多 3 个）:", flush=True)
            for p in test_paths[:_show_t]:
                print(f"  {p}", flush=True)

    train_data = [load_timing_graph(str(p)) for p in train_paths]
    val_data = [load_timing_graph(str(p)) for p in val_paths_list]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型骨干: {args.model_type}（hidden={args.hidden}, layers={args.layers}）", flush=True)
    if args.model_type in HETERO_CONV_MODELS:
        conv_cls, model_class_name = HETERO_CONV_MODELS[args.model_type]
        model = conv_cls(hidden_dim=args.hidden, num_layers=args.layers).to(device)
    elif args.model_type == "mpnn_mh":
        model = HeteroTimingMPNNMultiHop(hidden_dim=args.hidden, num_layers=args.layers).to(device)
        model_class_name = "HeteroTimingMPNNMultiHop"
    else:
        model = HeteroTimingMPNN(hidden_dim=args.hidden, num_layers=args.layers).to(device)
        model_class_name = "HeteroTimingMPNN"
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False)

    best_tau = float("-inf")
    save_path = Path(args.save)
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optimizer, device, graph_loss_weight=args.graph_loss_weight
        )
        metrics = eval_epoch(model, val_loader, device)
        tau = metrics["tau"]
        print(
            f"Epoch {epoch:4d}  train_loss={tr_loss:.6f}  "
            f"val_mae={metrics['mae']:.4f}  val_mape={metrics['mape']:.2f}%  "
            f"val_r2={metrics['r2']:.4f}  val_tau={metrics['tau']:.4f}  "
            f"val_spearman={metrics['spearman_r']:.4f}"
        )
        improved = not math.isnan(tau) and tau > best_tau
        if improved:
            best_tau = tau
            epochs_no_improve = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "model_class": model_class_name,
                    "graph_loss_weight": args.graph_loss_weight,
                },
                save_path,
            )
            print(f"  -> 保存最优模型 (tau={best_tau:.6f}) -> {save_path}")
        else:
            epochs_no_improve += 1

        if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
            print(
                f"早停：验证 tau 已连续 {args.patience} 个 epoch 未提升 "
                f"（当前第 {epoch} epoch，min_epochs={args.min_epochs}）。",
                flush=True,
            )
            break

    if save_path.is_file():
        try:
            ckpt = torch.load(save_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print(
            f"警告：未找到已保存的最优模型 {save_path}，以下汇总使用当前权重。",
            flush=True,
        )

    print("\n======== 最优模型 · 各划分节点级指标（归一化到达时间）========", flush=True)
    train_metrics = eval_epoch(model, train_loader, device)
    print(f"[Train] {format_metrics_line(train_metrics)}", flush=True)
    val_metrics = eval_epoch(model, val_loader, device)
    print(f"[Val]   {format_metrics_line(val_metrics)}", flush=True)

    if test_paths:
        test_data = [load_timing_graph(str(p)) for p in test_paths]
        test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False)
        test_metrics = eval_epoch(model, test_loader, device)
        print(f"[Test]  {format_metrics_line(test_metrics)}", flush=True)


if __name__ == "__main__":
    main()
