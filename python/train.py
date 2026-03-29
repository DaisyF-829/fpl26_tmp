"""
训练 TimingMPNN：递归加载 timing_graph.npz，MSE（y_valid 掩码），按验证集 Kendall τ 保存最优模型。
标签与掩码由 data_loader 根据 tnode_rt_time 与 tnode_valid_mask 构建。
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kendalltau
from torch_geometric.loader import DataLoader

from data_loader import load_timing_graph
from model import TimingMPNN


def _find_npz_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    if not root.is_dir():
        return paths
    for p in root.rglob("timing_graph.npz"):
        paths.append(p)
    return sorted(paths)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    crit = nn.MSELoss(reduction="sum")
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        mask = batch.y_valid
        if not mask.any():
            continue
        loss = crit(pred[mask], batch.y_arrival[mask])
        n = int(mask.sum().item())
        loss = loss / n
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * n
        total_count += n
    if total_count == 0:
        return 0.0
    return total_loss / total_count


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    abs_sum = 0.0
    sq_sum = 0.0
    n_mae = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            mask = batch.y_valid
            if not mask.any():
                continue
            p = pred[mask].detach().cpu().numpy()
            t = batch.y_arrival[mask].detach().cpu().numpy()
            preds.append(p)
            targets.append(t)
            abs_sum += float(np.abs(p - t).sum())
            sq_sum += float(((p - t) ** 2).sum())
            n_mae += p.size

    if n_mae == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "tau": float("nan")}

    mae = abs_sum / n_mae
    rmse = math.sqrt(sq_sum / n_mae)
    p_all = np.concatenate(preds)
    t_all = np.concatenate(targets)
    try:
        tau_stat = kendalltau(p_all, t_all, nan_policy="omit")
    except TypeError:
        tau_stat = kendalltau(p_all, t_all)
    corr = tau_stat.correlation
    tau = float(corr) if corr is not None and not math.isnan(float(corr)) else float("nan")

    return {"mae": mae, "rmse": rmse, "tau": tau}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--save", type=str, default="model.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_dir)
    all_paths = _find_npz_files(data_root)
    if not all_paths:
        raise SystemExit(f"在 {data_root} 下未找到 timing_graph.npz")

    if args.val_dir:
        val_root = Path(args.val_dir)
        val_paths_list = _find_npz_files(val_root)
        if not val_paths_list:
            raise SystemExit(f"在 {val_root} 下未找到 timing_graph.npz")
        train_paths = all_paths
        if not train_paths:
            raise SystemExit(f"在 {data_root} 下未找到 timing_graph.npz")
    else:
        idx = list(range(len(all_paths)))
        random.shuffle(idx)
        n_val = max(1, int(round(0.2 * len(all_paths))))
        val_idx = set(idx[:n_val])
        train_paths = [all_paths[i] for i in idx if i not in val_idx]
        val_paths_list = [all_paths[i] for i in idx if i in val_idx]
        if not train_paths:
            train_paths, val_paths_list = val_paths_list[:-1], val_paths_list[-1:]

    train_data = [load_timing_graph(str(p)) for p in train_paths]
    val_data = [load_timing_graph(str(p)) for p in val_paths_list]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimingMPNN(hidden_dim=args.hidden, num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False)

    best_tau = float("-inf")
    save_path = Path(args.save)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = eval_epoch(model, val_loader, device)
        tau = metrics["tau"]
        print(
            f"Epoch {epoch:4d}  train_loss={tr_loss:.6f}  "
            f"val_mae={metrics['mae']:.6f}  val_rmse={metrics['rmse']:.6f}  val_tau={tau:.6f}"
        )
        if not math.isnan(tau) and tau > best_tau:
            best_tau = tau
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": args.hidden,
                    "layers": args.layers,
                },
                save_path,
            )
            print(f"  -> 保存最优模型 (tau={best_tau:.6f}) -> {save_path}")


if __name__ == "__main__":
    main()
