from __future__ import annotations

import numpy as np
import pandas as pd


def format_break_label(left: float, right: float, suffix: str = "") -> str:
    """Format a numeric bin label with compact decimal places."""
    if np.isfinite(left) and np.isfinite(right):
        if abs(left - round(left)) < 0.05 and abs(right - round(right)) < 0.05:
            return f"{int(round(left))}-{int(round(right))}{suffix}"
        return f"{left:.1f}-{right:.1f}{suffix}"
    return f"{left:g}-{right:g}{suffix}"


def natural_break_edges(values: pd.Series | np.ndarray, n_classes: int = 5) -> list[float]:
    """
    Return data-driven class edges for one-dimensional values.

    This is a lightweight Fisher-Jenks implementation. Values are rounded to
    0.1 units and counted first, which keeps the dynamic program stable and fast
    for landslide-level slope values while preserving interpretable breakpoints.
    """
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().astype(float).to_numpy()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [0.0, 1.0]

    rounded = np.round(vals, 1)
    unique, counts = np.unique(rounded, return_counts=True)
    k = int(max(1, min(n_classes, unique.size)))
    if k <= 1:
        pad = max(abs(float(unique[0])) * 0.01, 0.5)
        return [float(unique[0] - pad), float(unique[0] + pad)]

    x = unique.astype(float)
    w = counts.astype(float)
    m = len(x)
    prefix_w = np.concatenate([[0.0], np.cumsum(w)])
    prefix_x = np.concatenate([[0.0], np.cumsum(w * x)])
    prefix_x2 = np.concatenate([[0.0], np.cumsum(w * x * x)])

    def segment_sse(i: int, j: int) -> float:
        weight = prefix_w[j] - prefix_w[i]
        if weight <= 0:
            return 0.0
        sx = prefix_x[j] - prefix_x[i]
        sx2 = prefix_x2[j] - prefix_x2[i]
        return float(max(sx2 - (sx * sx) / weight, 0.0))

    dp = np.full((k + 1, m + 1), np.inf)
    back = np.zeros((k + 1, m + 1), dtype=int)
    dp[0, 0] = 0.0
    for cls in range(1, k + 1):
        for end in range(cls, m + 1):
            best_cost = np.inf
            best_start = cls - 1
            for start in range(cls - 1, end):
                cost = dp[cls - 1, start] + segment_sse(start, end)
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
            dp[cls, end] = best_cost
            back[cls, end] = best_start

    starts = [m]
    end = m
    for cls in range(k, 0, -1):
        start = int(back[cls, end])
        starts.append(start)
        end = start
    starts = sorted(set(starts))

    edges = [float(vals.min())]
    for idx in starts[1:-1]:
        left = x[idx - 1]
        right = x[idx]
        edges.append(float((left + right) / 2.0))
    edges.append(float(vals.max()))

    edges = sorted(set(round(float(v), 6) for v in edges if np.isfinite(v)))
    if len(edges) < 2:
        pad = max(abs(float(vals[0])) * 0.01, 0.5)
        edges = [float(vals[0] - pad), float(vals[0] + pad)]
    return edges


def assign_natural_break_bins(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    left_col: str,
    right_col: str,
    order_col: str | None = None,
    n_classes: int = 5,
    suffix: str = "",
) -> pd.DataFrame:
    """Attach natural-break bin labels, left/right edges and optional order."""
    out = df.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    edges = natural_break_edges(values, n_classes=n_classes)
    labels = [format_break_label(edges[i], edges[i + 1], suffix=suffix) for i in range(len(edges) - 1)]
    label_to_left = {label: edges[i] for i, label in enumerate(labels)}
    label_to_right = {label: edges[i + 1] for i, label in enumerate(labels)}
    label_to_order = {label: i for i, label in enumerate(labels)}

    out[label_col] = pd.cut(values, bins=edges, labels=labels, include_lowest=True, right=True).astype(object)
    out[left_col] = out[label_col].astype(object).map(label_to_left)
    out[right_col] = out[label_col].astype(object).map(label_to_right)
    if order_col:
        out[order_col] = out[label_col].astype(object).map(label_to_order).fillna(999).astype(int)
    return out
