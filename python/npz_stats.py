"""
读取 timing_graph .npz，打印各数组形状、dtype 及统计量，便于核对导出数据。
不依赖 PyTorch / torch_geometric，仅需 numpy。

用法:
  python npz_stats.py path/to/file.npz
  python npz_stats.py path/to/dir              # 递归所有 .npz
  python npz_stats.py path/to/dir --max-files 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _fmt_float(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    ax = abs(x)
    if ax > 0 and (ax < 1e-4 or ax >= 1e6):
        return f"{x:.6e}"
    return f"{x:.6g}"


def _array_stats(name: str, a: np.ndarray) -> None:
    flat = np.asarray(a).ravel()
    n = flat.size
    print(f"  [{name}] shape={a.shape} dtype={a.dtype} size={n}", flush=True)
    if n == 0:
        return
    # 整数也按浮点算分位数，便于看范围
    f = flat.astype(np.float64, copy=False)
    nan_c = int(np.isnan(f).sum()) if np.issubdtype(f.dtype, np.floating) else 0
    fin = f[np.isfinite(f)] if nan_c or np.isinf(f).any() else f
    if fin.size == 0:
        print(f"      (无非有限值; nan={nan_c})", flush=True)
        return
    print(
        f"      min={_fmt_float(float(fin.min()))} max={_fmt_float(float(fin.max()))} "
        f"mean={_fmt_float(float(fin.mean()))} std={_fmt_float(float(fin.std()))}",
        flush=True,
    )
    if nan_c:
        print(f"      nan_count={nan_c}", flush=True)


def _bool_mask_stats(name: str, a: np.ndarray) -> None:
    b = np.asarray(a).astype(bool).ravel()
    n = b.size
    t = int(b.sum())
    print(f"  [{name}] shape={a.shape} True={t}/{n} ({100.0 * t / n:.2f}%)" if n else f"  [{name}] empty", flush=True)


def _try_get_array(z, key: str) -> np.ndarray | None:
    """npz 为懒加载；部分 C++ 导出的 dtype（如 descr '<?1'）当前 NumPy 无法解析，此时跳过该键。"""
    try:
        return z[key]
    except Exception as ex:
        print(
            f"  [{key}] 无法读取（跳过统计）: {ex}",
            flush=True,
        )
        print(
            "      提示: 多为导出端 bool/void 等与当前 NumPy 不兼容；可在导出侧改为 uint8/int8 或升级 NumPy。",
            flush=True,
        )
        return None


_MASK_KEYS = frozenset(
    {
        "tnode_pl_valid",
        "tnode_pl_arrival_mask",
        "tnode_valid_mask",
    }
)


def summarize_npz(path: Path) -> None:
    print(f"\n{'=' * 72}\n文件: {path.resolve()}\n{'=' * 72}", flush=True)
    z = np.load(path, allow_pickle=False)
    names = sorted(z.files)
    print(f"包含键 ({len(names)}): {', '.join(names)}", flush=True)

    for key in names:
        a = _try_get_array(z, key)
        if a is None:
            continue
        if a.size == 0:
            print(f"  [{key}] shape={a.shape} dtype={a.dtype} (空)", flush=True)
            continue
        # 布尔 / 整型小掩码：单独报 True 比例
        if key in _MASK_KEYS:
            _bool_mask_stats(key, a)
            continue
        if a.ndim <= 2 and a.size <= 10_000_000:
            _array_stats(key, a)
        else:
            print(f"  [{key}] shape={a.shape} dtype={a.dtype} size={a.size} (略过逐元素统计，过大)", flush=True)

    # 与 load_timing_graph (HeteroData) 对齐：全 N 节点，y_valid = mask & rt
    tnode_type = _try_get_array(z, "tnode_type") if "tnode_type" in z.files else None
    if tnode_type is not None:
        n = int(np.asarray(tnode_type).reshape(-1).shape[0])
        print(f"\n--- 训练标签摘要 (N={n}，与 npz 节点一一对应) ---", flush=True)
        if "tnode_valid_mask" in z.files:
            m = np.asarray(z["tnode_valid_mask"]).reshape(-1)[:n]
            vm = m.astype(bool) if m.dtype == np.bool_ else (m.astype(np.float32) != 0)
        else:
            vm = np.ones(n, dtype=bool)
        rt_time = _try_get_array(z, "tnode_rt_time") if "tnode_rt_time" in z.files else None
        if rt_time is not None:
            rt = np.asarray(rt_time, dtype=np.float64).reshape(-1)[:n]
            yv = vm & np.isfinite(rt) & (rt >= 0.0)
            print(
                f"  y_valid (= tnode_valid_mask & rt>=0 & 有限): {int(yv.sum())}/{n}",
                flush=True,
            )
            if yv.any():
                sub = rt[yv]
                print(
                    f"  tnode_rt_time (监督子集): min={_fmt_float(float(sub.min()))} "
                    f"max={_fmt_float(float(sub.max()))} mean={_fmt_float(float(sub.mean()))}",
                    flush=True,
                )
        else:
            print("  (无 tnode_rt_time 或无法读取)", flush=True)
    if "tedge_src" in z.files:
        es = _try_get_array(z, "tedge_src")
        if es is not None:
            e = int(np.asarray(es).reshape(-1).shape[0])
            print(f"\n--- 图规模 ---\n  边数 E={e}", flush=True)

    z.close()


def _find_npz_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".npz":
        return [root]
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.npz"))


def main() -> None:
    ap = argparse.ArgumentParser(description="打印 timing_graph npz 内数组统计")
    ap.add_argument("path", type=str, help=".npz 文件或包含 .npz 的目录")
    ap.add_argument("--max-files", type=int, default=0, help="目录模式下最多处理多少个文件，0 表示不限制")
    args = ap.parse_args()

    root = Path(args.path)
    paths = _find_npz_files(root)
    if not paths:
        raise SystemExit(f"未找到 .npz: {root}")

    if args.max_files > 0:
        paths = paths[: args.max_files]

    for p in paths:
        try:
            summarize_npz(p)
        except Exception as ex:
            print(f"\n[错误] {p}: {ex}", flush=True)
            raise


if __name__ == "__main__":
    main()
