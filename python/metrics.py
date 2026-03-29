"""
节点级回归指标：MAE、RMSE、Kendall τ、Spearman ρ、MAPE（t>0.01）、R²。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


def compute_regression_metrics(p: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """p, t 为同长度一维数组（如归一化后的预测与 y_arrival）。"""
    p = np.asarray(p, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    n = int(p.size)
    nan = float("nan")
    if n == 0 or t.size != n:
        return {
            "mae": nan,
            "rmse": nan,
            "tau": nan,
            "spearman_r": nan,
            "mape": nan,
            "r2": nan,
        }

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    try:
        tau_stat = kendalltau(p, t, nan_policy="omit")
    except TypeError:
        tau_stat = kendalltau(p, t)
    tc: Any = tau_stat.correlation
    tau = float(tc) if tc is not None and not math.isnan(float(tc)) else nan

    try:
        sp_stat = spearmanr(p, t, nan_policy="omit")
    except TypeError:
        sp_stat = spearmanr(p, t)
    sc: Any = getattr(sp_stat, "correlation", None)
    if sc is None:
        sc = getattr(sp_stat, "statistic", None)
    if sc is None or (isinstance(sc, float) and math.isnan(float(sc))):
        spearman_r = nan
    else:
        spearman_r = float(sc)

    mape_mask = t > 0.01
    if int(mape_mask.sum()) > 0:
        mape = float(
            np.mean(np.abs(p[mape_mask] - t[mape_mask]) / t[mape_mask]) * 100.0
        )
    else:
        mape = nan

    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan

    return {
        "mae": mae,
        "rmse": rmse,
        "tau": tau,
        "spearman_r": spearman_r,
        "mape": mape,
        "r2": r2,
    }


def format_metrics_line(m: dict[str, float]) -> str:
    """单行打印用。"""

    def fmt(key: str, spec: str) -> str:
        v = m.get(key, float("nan"))
        if isinstance(v, float) and math.isnan(v):
            return "nan"
        return format(v, spec)

    return (
        f"tau={fmt('tau', '.4f')}  spearman={fmt('spearman_r', '.4f')}  "
        f"r2={fmt('r2', '.4f')}  mape={fmt('mape', '.2f')}%  "
        f"mae={fmt('mae', '.4f')}  rmse={fmt('rmse', '.4f')}"
    )
