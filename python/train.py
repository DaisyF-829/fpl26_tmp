"""
训练 HeteroTimingMPNN：HeteroData（4 类边），MSE（tnode.y_valid 掩码），按验证集 Kendall τ 保存最优模型。
默认从 data_dir 按 npz 文件三路划分：先 test，再 val，其余 train（无重叠）。
--val_dir：验证集改来自单独目录（忽略 val_frac）；--test_dir：测试集改来自单独目录（忽略 test_frac）。
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
from scipy.stats import kendalltau
from torch_geometric.loader import DataLoader

from data_loader import load_timing_graph
from model import HeteroTimingMPNN


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


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    crit = nn.MSELoss(reduction="sum")
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        mask = batch["tnode"].y_valid
        if not mask.any():
            continue
        loss = crit(pred[mask], batch["tnode"].y_arrival[mask])
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
            pred = model(batch)
            mask = batch["tnode"].y_valid
            if not mask.any():
                continue
            p = pred[mask].detach().cpu().numpy()
            t = batch["tnode"].y_arrival[mask].detach().cpu().numpy()
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
    model = HeteroTimingMPNN(hidden_dim=args.hidden, num_layers=args.layers).to(device)
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
                    "model_class": "HeteroTimingMPNN",
                },
                save_path,
            )
            print(f"  -> 保存最优模型 (tau={best_tau:.6f}) -> {save_path}")

    if test_paths:
        test_data = [load_timing_graph(str(p)) for p in test_paths]
        test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False)
        if save_path.is_file():
            try:
                ckpt = torch.load(save_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        else:
            print(
                f"警告：未找到已保存的最优模型 {save_path}，使用当前权重评估测试集。",
                flush=True,
            )
        test_metrics = eval_epoch(model, test_loader, device)
        print(
            f"\n[Test]  mae={test_metrics['mae']:.6f}  "
            f"rmse={test_metrics['rmse']:.6f}  tau={test_metrics['tau']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
