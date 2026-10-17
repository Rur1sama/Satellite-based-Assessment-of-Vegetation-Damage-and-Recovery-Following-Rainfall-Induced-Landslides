from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .io_utils import ensure_dir, safe_to_csv


OUTCOME_COLORS = {
    "LAI_SR2": "#b2182b",
    "FVC": "#1b7837",
    "FAPAR": "#e08214",
    "GPP": "#2166ac",
    "VOD": "#6a3d9a",
}


OUTCOME_DENSITY_COLORS = {
    "LAI_SR2": [[0, "rgba(255,255,255,0)"], [0.35, "rgba(244,165,130,0.28)"], [1, "rgba(178,24,43,0.85)"]],
    "FVC": [[0, "rgba(255,255,255,0)"], [0.35, "rgba(166,219,160,0.28)"], [1, "rgba(27,120,55,0.85)"]],
    "FAPAR": [[0, "rgba(255,255,255,0)"], [0.35, "rgba(253,184,99,0.28)"], [1, "rgba(224,130,20,0.85)"]],
    "GPP": [[0, "rgba(255,255,255,0)"], [0.35, "rgba(146,197,222,0.28)"], [1, "rgba(33,102,172,0.85)"]],
    "VOD": [[0, "rgba(255,255,255,0)"], [0.35, "rgba(202,178,214,0.28)"], [1, "rgba(106,61,154,0.85)"]],
}

DENSITY_SCATTER_COLORSCALE = [
    [0.00, "#3f007d"],
    [0.18, "#2c7fb8"],
    [0.36, "#41b6c4"],
    [0.54, "#a1dab4"],
    [0.70, "#ffffbf"],
    [0.84, "#fdae61"],
    [1.00, "#b2182b"],
]

EVENT_STUDY_PLOT_RELATIVE_YEARS = list(range(-2, 6))
EVENT_STUDY_PERCENT_RANGE = [-80, 20]
EVENT_STUDY_PERCENT_TICKS = [-80, -60, -40, -20, 0, 20]


def _rgba(hex_color: str, alpha: float) -> str:
    """把 #RRGGBB 颜色转换成 Plotly 可用的 rgba 字符串。"""
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return f"rgba(80,80,80,{float(alpha)})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha)})"


def _mix_hex(hex_color: str, other: str = "#ffffff", ratio: float = 0.5) -> str:
    """把主色与另一种颜色混合，用来生成同一 outcome 的深浅线条。"""
    def parse(value: str) -> tuple[int, int, int]:
        text = str(value).lstrip("#")
        if len(text) != 6:
            return (80, 80, 80)
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))

    r1, g1, b1 = parse(hex_color)
    r2, g2, b2 = parse(other)
    ratio = min(max(float(ratio), 0.0), 1.0)
    r = round(r1 * (1.0 - ratio) + r2 * ratio)
    g = round(g1 * (1.0 - ratio) + g2 * ratio)
    b = round(b1 * (1.0 - ratio) + b2 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _pchip_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """计算形状保持 Hermite 曲线斜率，避免曲线越过相邻样本点极值。"""
    n = len(xs)
    if n <= 1:
        return [0.0] * n
    if n == 2:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
        return [slope, slope]
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] if h[i] != 0 else 0.0 for i in range(n - 1)]
    slopes = [0.0] * n

    def endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        if h0 == 0:
            return 0.0
        slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1) if (h0 + h1) != 0 else d0
        if slope * d0 <= 0:
            return 0.0
        if d0 * d1 < 0 and abs(slope) > abs(3 * d0):
            return 3 * d0
        return slope

    slopes[0] = endpoint_slope(h[0], h[1], delta[0], delta[1])
    slopes[-1] = endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            slopes[i] = 0.0
            continue
        w1 = 2 * h[i] + h[i - 1]
        w2 = h[i] + 2 * h[i - 1]
        denom = (w1 / delta[i - 1]) + (w2 / delta[i])
        slopes[i] = (w1 + w2) / denom if denom != 0 else 0.0
    return slopes


def _smooth_xy(
    x_values: pd.Series,
    y_values: pd.Series,
    points_per_segment: int = 16,
) -> tuple[list[float], list[float]]:
    """生成经过原始节点且不制造额外极值的平滑点线。"""
    curve = (
        pd.DataFrame({"x": pd.to_numeric(x_values, errors="coerce"), "y": pd.to_numeric(y_values, errors="coerce")})
        .dropna()
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
    )
    if len(curve) <= 2:
        return curve["x"].astype(float).tolist(), curve["y"].astype(float).tolist()
    xs = curve["x"].astype(float).tolist()
    ys = curve["y"].astype(float).tolist()
    slopes = _pchip_slopes(xs, ys)
    smooth_x: list[float] = []
    smooth_y: list[float] = []
    steps = max(int(points_per_segment), 2)
    for i in range(len(xs) - 1):
        h = xs[i + 1] - xs[i]
        if h == 0:
            continue
        for step in range(steps + 1):
            if i > 0 and step == 0:
                continue
            t = step / steps
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            y = h00 * ys[i] + h10 * h * slopes[i] + h01 * ys[i + 1] + h11 * h * slopes[i + 1]
            lo = min(ys[i], ys[i + 1])
            hi = max(ys[i], ys[i + 1])
            smooth_x.append(xs[i] + t * h)
            smooth_y.append(min(max(y, lo), hi))
    return smooth_x, smooth_y


def _ordered_outcomes(values: list[str]) -> list[str]:
    """按照主图常用顺序排列生态指标。"""
    preferred = ["FVC", "LAI_SR2", "GPP"]
    allowed = set(preferred)
    rank = {name: idx for idx, name in enumerate(preferred)}
    return sorted([str(v) for v in values if str(v) in allowed], key=lambda x: (rank.get(x, 999), x))


def _series_label(series: str) -> str:
    """把大小分组名称转为图例文本。"""
    labels = {
        "area_weighted_all": "area-weighted all",
        "small": "small (<1000 m2)",
        "large": "large (>=1000 m2)",
    }
    return labels.get(str(series), str(series))


def _size_classes_in_order(values: pd.Series | list[str]) -> list[str]:
    """返回大小滑坡分组顺序；主口径只使用 small / large。"""
    raw = pd.Series(values).dropna().astype(str).unique().tolist()
    preferred = [name for name in ["small", "large"] if name in raw]
    if preferred:
        return preferred
    legacy = [name for name in ["no_inner_small", "inner_available_large"] if name in raw]
    return legacy or sorted(raw)


def _fixed_area_bins_labels() -> tuple[list[float], list[str]]:
    """统一定义滑坡面积等级，供统计表和图件共用。"""
    bins = [0, 200, 500, 1000, 2000, 5000, np.inf]
    labels = [
        "100-200",
        "200-500",
        "500-1000",
        "1000-2000",
        "2000-5000",
        ">5000",
    ]
    return bins, labels


def _axis_domain_ref(axis: str, subplot_index: int) -> str:
    """Plotly 第一个子图坐标轴不带数字，其余子图从 2 开始编号。"""
    return f"{axis} domain" if subplot_index == 1 else f"{axis}{subplot_index} domain"


def _write_html_with_fallback(fig: go.Figure, path: Path) -> Path:
    """写 HTML；目标文件被占用时自动创建不会覆盖的新文件。"""
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    candidates = [
        path,
        path.with_name(f"{path.stem}_latest{path.suffix}"),
        path.with_name(f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"),
    ]
    errors: list[str] = []
    for candidate in candidates:
        try:
            candidate.write_text(html, encoding="utf-8")
            return candidate
        except PermissionError as exc:
            errors.append(f"{candidate}: {exc}")
    raise PermissionError("; ".join(errors))


def _add_caption(fig: go.Figure, text: str, y: float = -0.16) -> None:
    """在 HTML 图底部添加论文图注说明。"""
    fig.add_annotation(
        text=str(text),
        x=0,
        y=float(y),
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#222222"),
    )


def _se_from_ci(row: pd.Series) -> float:
    """从 95% CI 近似还原标准误。"""
    low = pd.to_numeric(row.get("ci_low_percent"), errors="coerce")
    high = pd.to_numeric(row.get("ci_high_percent"), errors="coerce")
    if pd.isna(low) or pd.isna(high):
        return np.nan
    return float(abs(high - low) / (2.0 * 1.96))


def _normal_p_from_diff(mean_a: float, se_a: float, mean_b: float, se_b: float) -> tuple[float, float]:
    """用正态近似检验大小滑坡 ATT% 差异。"""
    if any(pd.isna(x) for x in [mean_a, se_a, mean_b, se_b]):
        return np.nan, np.nan
    se = math.sqrt(float(se_a) ** 2 + float(se_b) ** 2)
    if se <= 0:
        return np.nan, np.nan
    z = (float(mean_b) - float(mean_a)) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def _point_density(x: pd.Series, y: pd.Series, bins: int = 55) -> np.ndarray:
    """用二维频率格给每个点分配密度，速度比 KDE 更稳定。"""
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    density = np.full(len(x_arr), np.nan, dtype=float)
    if valid.sum() < 5:
        return density
    xv = x_arr[valid]
    yv = y_arr[valid]
    hist, x_edges, y_edges = np.histogram2d(xv, yv, bins=int(bins))
    x_bin = np.clip(np.searchsorted(x_edges, xv, side="right") - 1, 0, hist.shape[0] - 1)
    y_bin = np.clip(np.searchsorted(y_edges, yv, side="right") - 1, 0, hist.shape[1] - 1)
    counts = hist[x_bin, y_bin]
    density[valid] = counts / max(float(valid.sum()), 1.0)
    return density


def _smooth2d(values: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """用轻量高斯核平滑二维频率面，避免依赖 KDE 或 scipy。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return arr
    radius = max(1, int(round(float(sigma) * 3)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=arr)
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=smoothed)
    return smoothed


def _density_surface(
    x: pd.Series,
    y: pd.Series,
    x_bins: int = 90,
    y_bins: int = 80,
    y_range: tuple[float, float] = (0.0, 100.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把面积-恢复率点云转换成平滑二维频率面，用于绘制连续密度底图。"""
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0)
    if valid.sum() < 5:
        return np.array([]), np.array([]), np.empty((0, 0), dtype=float)
    xv = np.log10(x_arr[valid])
    yv = np.clip(y_arr[valid], y_range[0], y_range[1])
    x_min = float(np.nanmin(xv))
    x_max = float(np.nanmax(xv))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        return np.array([]), np.array([]), np.empty((0, 0), dtype=float)
    hist, x_edges, y_edges = np.histogram2d(
        xv,
        yv,
        bins=[int(x_bins), int(y_bins)],
        range=[[x_min, x_max], [float(y_range[0]), float(y_range[1])]],
    )
    z = _smooth2d(hist.T / max(float(valid.sum()), 1.0), sigma=1.7)
    z[z <= 0] = np.nan
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return 10.0 ** x_centers, y_centers, z


def _simple_linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """不用 np.polyfit，直接用协方差公式计算一元线性拟合，避免底层线代库不稳定。"""
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3:
        return np.nan, np.nan
    xv = x_arr[valid]
    yv = y_arr[valid]
    x_mean = float(xv.mean())
    y_mean = float(yv.mean())
    denom = float(((xv - x_mean) ** 2).sum())
    if denom <= 0:
        return np.nan, np.nan
    slope = float(((xv - x_mean) * (yv - y_mean)).sum() / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _weighted_all_att_percent(class_df: pd.DataFrame) -> pd.DataFrame:
    """按滑坡面积把 small/large 的 DID ATT% 合成为总体曲线。"""
    rows: list[dict[str, object]] = []
    for (outcome, zone_name, rel), chunk in class_df.groupby(["outcome", "zone_name", "relative_year"], dropna=True):
        weights = pd.to_numeric(chunk["total_area_m2"], errors="coerce").where(lambda s: s > 0)
        if weights.notna().sum() == 0:
            continue
        weights = weights.fillna(weights.dropna().median())
        total_w = float(weights.sum())
        if "n_landslides" in chunk.columns:
            n_landslides = int(pd.to_numeric(chunk["n_landslides"], errors="coerce").fillna(0).sum())
        elif "n_treated" in chunk.columns:
            n_landslides = int(pd.to_numeric(chunk["n_treated"], errors="coerce").fillna(0).sum())
        else:
            n_landslides = 0
        row = {
            "outcome": outcome,
            "zone_name": zone_name,
            "relative_year": int(rel),
            "series": "area_weighted_all",
            "att_percent": float((chunk["att_percent"] * weights).sum() / total_w),
            "ci_low_percent": float((chunk["ci_low_percent"] * weights).sum() / total_w),
            "ci_high_percent": float((chunk["ci_high_percent"] * weights).sum() / total_w),
            "se": np.nan,
            "n_landslides": n_landslides,
            "total_area_m2": total_w,
        }
        row["se"] = float(abs(row["ci_high_percent"] - row["ci_low_percent"]) / (2.0 * 1.96))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_individual_att_percent(
    panel: pd.DataFrame,
    group_time: pd.DataFrame,
    assignment: pd.DataFrame,
    zone_name: str,
    value_column: str = "mean",
    reference_relative_year: int = -1,
) -> pd.DataFrame:
    """给每个滑坡单独计算 DID ATT%，用于面积-恢复关系散点图。"""
    required_panel = {
        "outcome",
        "event_id",
        "event_year",
        "relative_year",
        "zone_name",
        "zone_role",
        "area_m2",
        value_column,
    }
    required_group = {"outcome", "zone_name", "size_class", "cohort_year", "relative_year", "control_change"}
    required_assign = {"event_id", "size_class", "landslide_core_area_m2"}
    if panel.empty or group_time.empty or assignment.empty:
        return pd.DataFrame()
    if not required_panel.issubset(panel.columns):
        return pd.DataFrame()
    if not required_group.issubset(group_time.columns):
        return pd.DataFrame()
    if not required_assign.issubset(assignment.columns):
        return pd.DataFrame()

    treated = panel[
        (panel["zone_role"].astype(str) == "treated") & (panel["zone_name"].astype(str) == str(zone_name))
    ].copy()
    if treated.empty:
        return pd.DataFrame()
    treated["event_year"] = pd.to_numeric(treated["event_year"], errors="coerce")
    treated["relative_year"] = pd.to_numeric(treated["relative_year"], errors="coerce")
    treated[value_column] = pd.to_numeric(treated[value_column], errors="coerce")
    treated["area_m2"] = pd.to_numeric(treated["area_m2"], errors="coerce")
    treated = treated.dropna(subset=["event_year", "relative_year", value_column])
    if treated.empty:
        return pd.DataFrame()

    assign = assignment[["event_id", "size_class", "landslide_core_area_m2"]].drop_duplicates("event_id").copy()
    assign["landslide_core_area_m2"] = pd.to_numeric(assign["landslide_core_area_m2"], errors="coerce")
    treated = treated.drop(columns=[col for col in ["size_class", "landslide_core_area_m2"] if col in treated.columns])
    treated = treated.merge(assign, on="event_id", how="left")
    treated = treated.dropna(subset=["size_class", "landslide_core_area_m2"]).copy()
    if treated.empty:
        return pd.DataFrame()

    baseline = treated[treated["relative_year"] == int(reference_relative_year)][
        ["outcome", "event_id", value_column]
    ].rename(columns={value_column: "baseline_value"})
    treated = treated.merge(baseline, on=["outcome", "event_id"], how="left")
    treated = treated.dropna(subset=["baseline_value"]).copy()
    if treated.empty:
        return pd.DataFrame()

    controls = group_time.copy()
    controls = controls[controls["zone_name"].astype(str) == str(zone_name)].copy()
    controls["cohort_year"] = pd.to_numeric(controls["cohort_year"], errors="coerce")
    controls["relative_year"] = pd.to_numeric(controls["relative_year"], errors="coerce")
    controls["control_change"] = pd.to_numeric(controls["control_change"], errors="coerce")
    controls = controls.dropna(subset=["cohort_year", "relative_year", "control_change"])
    controls = controls[
        ["outcome", "size_class", "cohort_year", "relative_year", "control_change"]
    ].drop_duplicates(["outcome", "size_class", "cohort_year", "relative_year"])
    if controls.empty:
        return pd.DataFrame()

    result = treated.merge(
        controls,
        left_on=["outcome", "size_class", "event_year", "relative_year"],
        right_on=["outcome", "size_class", "cohort_year", "relative_year"],
        how="left",
    )
    result = result.dropna(subset=["control_change"]).copy()
    if result.empty:
        return pd.DataFrame()

    result["counterfactual_value"] = result["baseline_value"] + result["control_change"]
    result["individual_att"] = result[value_column] - result["counterfactual_value"]
    valid = result["counterfactual_value"] > 0
    result["individual_att_percent"] = np.nan
    result.loc[valid, "individual_att_percent"] = (
        result.loc[valid, "individual_att"] / result.loc[valid, "counterfactual_value"] * 100.0
    )
    result = result.dropna(subset=["individual_att_percent"]).copy()
    result["event_year"] = result["event_year"].astype(int)
    result["relative_year"] = result["relative_year"].astype(int)
    keep = [
        "outcome",
        "event_id",
        "event_year",
        "relative_year",
        "size_class",
        "landslide_core_area_m2",
        "area_m2",
        "baseline_value",
        value_column,
        "control_change",
        "counterfactual_value",
        "individual_att",
        "individual_att_percent",
    ]
    return result[keep].sort_values(["outcome", "event_id", "relative_year"])


def _individual_recovery_rows(individual_att: pd.DataFrame) -> pd.DataFrame:
    """把逐年 individual ATT% 曲线压缩为每个滑坡一个恢复指标点。"""
    if individual_att.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (outcome, event_id), curve in individual_att.groupby(["outcome", "event_id"], dropna=True):
        curve = curve.sort_values("relative_year").copy()
        event_row = curve[curve["relative_year"] == 0]
        post = curve[curve["relative_year"] > 0]
        if event_row.empty or post.empty:
            continue
        event_att = float(event_row.iloc[0]["individual_att_percent"])
        latest = post.iloc[-1]
        latest_year = int(latest["relative_year"])
        latest_att = float(latest["individual_att_percent"])
        year1 = curve[curve["relative_year"] == 1]
        year1_att = float(year1.iloc[0]["individual_att_percent"]) if not year1.empty else np.nan
        if event_att < 0 and latest_year > 0:
            latest_recovery = (latest_att - event_att) / (0.0 - event_att) * 100.0
            speed = latest_recovery / latest_year
            first_year_recovery = (year1_att - event_att) / (0.0 - event_att) * 100.0 if pd.notna(year1_att) else np.nan
        else:
            latest_recovery = np.nan
            speed = np.nan
            first_year_recovery = np.nan
        valid_recovery = bool(event_att <= -20.0 and pd.notna(first_year_recovery))
        first_year_recovery_capped = (
            float(np.clip(first_year_recovery, 0.0, 100.0)) if valid_recovery else np.nan
        )
        latest_recovery_capped = float(np.clip(latest_recovery, 0.0, 100.0)) if pd.notna(latest_recovery) else np.nan
        rows.append(
            {
                "outcome": outcome,
                "event_id": event_id,
                "event_year": int(event_row.iloc[0]["event_year"]),
                "size_class": str(event_row.iloc[0]["size_class"]),
                "landslide_core_area_m2": float(event_row.iloc[0]["landslide_core_area_m2"]),
                "event_att_percent": event_att,
                "year1_att_percent": year1_att,
                "first_year_recovery_percent_raw": first_year_recovery,
                "first_year_recovery_percent_capped": first_year_recovery_capped,
                "latest_relative_year": latest_year,
                "latest_att_percent": latest_att,
                "latest_recovery_percent_raw": latest_recovery,
                "latest_recovery_percent_capped": latest_recovery_capped,
                "recovery_speed_percent_per_year_raw": speed,
                "valid_recovery_scatter": valid_recovery,
                "years_to_50pct_recovery": _years_to_recovery_threshold(
                    curve.rename(columns={"individual_att_percent": "att_percent"}), event_att, 50
                ),
                "years_to_80pct_recovery": _years_to_recovery_threshold(
                    curve.rename(columns={"individual_att_percent": "att_percent"}), event_att, 80
                ),
                "years_to_95pct_recovery": _years_to_recovery_threshold(
                    curve.rename(columns={"individual_att_percent": "att_percent"}), event_att, 95
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "landslide_core_area_m2", "event_id"])


def _binned_recovery_trend(individual_recovery: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    """按面积分位数生成中位恢复速度趋势线，辅助观察可能的转折。"""
    if individual_recovery.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for outcome, chunk in individual_recovery.groupby("outcome", dropna=True):
        if "valid_recovery_scatter" not in chunk.columns:
            continue
        df = chunk[chunk["valid_recovery_scatter"].astype(bool)].copy()
        df = df.dropna(subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"])
        df = df[df["landslide_core_area_m2"] > 0]
        if len(df) < 4:
            continue
        bins = min(int(n_bins), max(2, df["landslide_core_area_m2"].nunique()))
        try:
            df["area_bin"] = pd.qcut(df["landslide_core_area_m2"], q=bins, duplicates="drop")
        except ValueError:
            continue
        for _, bdf in df.groupby("area_bin", observed=True):
            rows.append(
                {
                    "outcome": outcome,
                    "n_events": int(len(bdf)),
                    "area_median_m2": float(bdf["landslide_core_area_m2"].median()),
                    "area_min_m2": float(bdf["landslide_core_area_m2"].min()),
                    "area_max_m2": float(bdf["landslide_core_area_m2"].max()),
                    "first_year_recovery_median_percent": float(
                        bdf["first_year_recovery_percent_capped"].median()
                    ),
                    "first_year_recovery_q25_percent": float(bdf["first_year_recovery_percent_capped"].quantile(0.25)),
                    "first_year_recovery_q75_percent": float(bdf["first_year_recovery_percent_capped"].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "area_median_m2"])


def _fixed_area_bin_summary(individual_recovery: pd.DataFrame) -> pd.DataFrame:
    """按固定面积等级汇总首年恢复率和 over-recovery 比例。"""
    if individual_recovery.empty or "valid_recovery_scatter" not in individual_recovery.columns:
        return pd.DataFrame()
    df = individual_recovery[individual_recovery["valid_recovery_scatter"].astype(bool)].copy()
    df = df.dropna(subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"])
    df = df[df["landslide_core_area_m2"] > 0].copy()
    if df.empty:
        return pd.DataFrame()
    bins, labels = _fixed_area_bins_labels()
    df["area_bin_label"] = pd.cut(df["landslide_core_area_m2"], bins=bins, labels=labels, include_lowest=True)
    rows: list[dict[str, object]] = []
    for (outcome, area_bin), chunk in df.dropna(subset=["area_bin_label"]).groupby(
        ["outcome", "area_bin_label"], observed=True
    ):
        raw = pd.to_numeric(chunk["first_year_recovery_percent_raw"], errors="coerce")
        capped = pd.to_numeric(chunk["first_year_recovery_percent_capped"], errors="coerce")
        rows.append(
            {
                "outcome": outcome,
                "area_bin_label": str(area_bin),
                "area_bin_order": labels.index(str(area_bin)),
                "n_events": int(len(chunk)),
                "area_median_m2": float(chunk["landslide_core_area_m2"].median()),
                "recovery_median_percent": float(capped.median()),
                "recovery_q25_percent": float(capped.quantile(0.25)),
                "recovery_q75_percent": float(capped.quantile(0.75)),
                "over_recovery_share_percent": float((raw > 100).mean() * 100.0),
                "no_recovery_share_percent": float((raw <= 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "area_bin_order"])


def plot_area_recovery_joint_summary(
    individual_recovery: pd.DataFrame,
    bin_summary: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """生成论文式面积-恢复联合分布图。"""
    figures_dir = ensure_dir(figures_dir)
    if individual_recovery.empty or bin_summary.empty:
        return []
    df = individual_recovery[
        individual_recovery.get("valid_recovery_scatter", pd.Series(False, index=individual_recovery.index)).astype(bool)
    ].copy()
    df = df.dropna(subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"])
    df = df[df["landslide_core_area_m2"] > 0].copy()
    if df.empty:
        return []
    bins, labels = _fixed_area_bins_labels()
    df["area_bin_label"] = pd.cut(df["landslide_core_area_m2"], bins=bins, labels=labels, include_lowest=True)
    df["log10_area"] = np.log10(df["landslide_core_area_m2"].astype(float))
    outcomes = _ordered_outcomes(df["outcome"].dropna().unique().tolist())
    rows = len(outcomes)
    fig = make_subplots(rows=rows, cols=3, horizontal_spacing=0.08, vertical_spacing=0.10)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    letter_idx = 0
    area_ticks = [100, 300, 1000, 3000, 10000, 30000, 100000]

    for row, outcome in enumerate(outcomes, start=1):
        color = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        odf = df[df["outcome"] == outcome].copy()
        sdf = bin_summary[bin_summary["outcome"] == outcome].copy()
        if odf.empty or sdf.empty:
            continue

        density_x, density_y, density_z = _density_surface(
            odf["landslide_core_area_m2"],
            odf["first_year_recovery_percent_capped"],
        )
        if density_z.size:
            fig.add_trace(
                go.Heatmap(
                    x=density_x,
                    y=density_y,
                    z=density_z,
                    colorscale=DENSITY_SCATTER_COLORSCALE,
                    zsmooth="best",
                    opacity=0.88,
                    showscale=(row == 1),
                    colorbar=(
                        dict(
                            title="frequency",
                            len=0.26,
                            y=0.88,
                            x=1.035,
                            thickness=18,
                            outlinewidth=0.8,
                        )
                        if row == 1
                        else None
                    ),
                    showlegend=False,
                    hovertemplate="frequency=%{z:.4f}<br>area=%{x:.0f} m2<br>recovery=%{y:.1f}%<extra></extra>",
                ),
                row=row,
                col=1,
            )
        odf["density"] = _point_density(odf["log10_area"], odf["first_year_recovery_percent_capped"])
        odf["density"] = pd.to_numeric(odf["density"], errors="coerce").fillna(0.0)
        odf = odf.sort_values("density", na_position="first").copy()
        fig.add_trace(
            go.Scattergl(
                x=odf["landslide_core_area_m2"],
                y=odf["first_year_recovery_percent_capped"],
                mode="markers",
                marker=dict(
                    color="rgba(63,0,125,0.58)",
                    opacity=0.62,
                    size=3.0,
                    line=dict(width=0),
                ),
                showlegend=False,
                customdata=np.stack(
                    [
                        odf["landslide_core_area_m2"],
                        odf["first_year_recovery_percent_raw"],
                        odf["density"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "area=%{customdata[0]:.0f} m2<br>"
                    "first-year recovery=%{y:.1f}%<br>"
                    "raw recovery=%{customdata[1]:.1f}%<br>"
                    "frequency=%{customdata[2]:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        fit_df = odf.dropna(subset=["log10_area", "first_year_recovery_percent_capped"]).copy()
        if len(fit_df) >= 3 and fit_df["log10_area"].nunique() > 1:
            slope, intercept = _simple_linear_fit(
                fit_df["log10_area"],
                fit_df["first_year_recovery_percent_capped"],
            )
            if not np.isfinite(slope) or not np.isfinite(intercept):
                continue
            x_fit = np.linspace(float(fit_df["log10_area"].min()), float(fit_df["log10_area"].max()), 80)
            y_fit = slope * x_fit + intercept
            fig.add_trace(
                go.Scatter(
                    x=10.0 ** x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color="blue", width=2.2),
                    name="linear fit",
                    showlegend=(row == 1),
                    hovertemplate="fitted trend<br>area=%{x:.0f} m2<br>recovery=%{y:.1f}%<extra></extra>",
                ),
                row=row,
                col=1,
            )

        for label in labels:
            bdf = odf[odf["area_bin_label"].astype(str) == label]
            if bdf.empty:
                continue
            fig.add_trace(
                go.Box(
                    x=[label] * len(bdf),
                    y=bdf["first_year_recovery_percent_capped"],
                    name=label,
                    marker=dict(color=color, opacity=0.30, size=2),
                    line=dict(color=color, width=1),
                    fillcolor="rgba(255,255,255,0)",
                    boxpoints=False,
                    showlegend=False,
                    hovertemplate=f"area bin={label}<br>recovery=%{{y:.1f}}%<extra></extra>",
                ),
                row=row,
                col=2,
            )

        fig.add_trace(
            go.Bar(
                x=sdf["area_bin_label"],
                y=sdf["over_recovery_share_percent"],
                marker=dict(color=color, opacity=0.72),
                name="over-recovery",
                showlegend=(row == 1),
                customdata=np.stack([sdf["n_events"], sdf["recovery_median_percent"]], axis=-1),
                hovertemplate="area bin=%{x}<br>over-recovery=%{y:.1f}%<br>n=%{customdata[0]}<br>median recovery=%{customdata[1]:.1f}%<extra></extra>",
            ),
            row=row,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=sdf["area_bin_label"],
                y=sdf["recovery_median_percent"],
                mode="lines+markers",
                line=dict(color="black", width=2),
                marker=dict(color="black", size=6),
                name="median recovery",
                showlegend=(row == 1),
                hovertemplate="area bin=%{x}<br>median recovery=%{y:.1f}%<extra></extra>",
            ),
            row=row,
            col=3,
        )

        for col in [1, 2, 3]:
            fig.add_annotation(
                text=f"({letters[letter_idx]})",
                x=0.01,
                y=0.95,
                xref=_axis_domain_ref("x", (row - 1) * 3 + col),
                yref=_axis_domain_ref("y", (row - 1) * 3 + col),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
            letter_idx += 1
        fig.update_xaxes(
            tickmode="array",
            tickvals=area_ticks,
            ticktext=[str(x) for x in area_ticks],
            type="log",
            row=row,
            col=1,
        )

    fig.update_layout(
        title=dict(
            text="Area-recovery joint summary",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial", color="#111111"),
        ),
        template="plotly_white",
        width=1700,
        height=max(520, 330 * rows + 150),
        margin=dict(l=85, r=230, t=78, b=150),
        legend=dict(x=1.04, y=0.54, xanchor="left", yanchor="middle", bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family="Arial", size=12, color="#111111"),
    )
    for row in range(1, rows + 1):
        fig.update_xaxes(title_text="Core area (m2, log scale)" if row == rows else "", showline=True, mirror=True, ticks="outside", row=row, col=1)
        fig.update_yaxes(title_text="First-year recovery (%)" if row == 2 or rows == 1 else "", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=1)
        fig.update_xaxes(title_text="Core area class (m2)" if row == rows else "", showline=True, mirror=True, ticks="outside", tickangle=35, row=row, col=2)
        fig.update_yaxes(title_text="", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=2)
        fig.update_xaxes(title_text="Core area class (m2)" if row == rows else "", showline=True, mirror=True, ticks="outside", tickangle=35, row=row, col=3)
        fig.update_yaxes(title_text="Share / median (%)" if row == 2 or rows == 1 else "", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=3)
    fig.add_annotation(
        text=(
            "Area-recovery joint summary in town Jiangwan. "
            "Left panels are exploratory density-scatter diagnostics for individual landslides, using core area on a log scale and first-year DID recovery capped to 0-100%. "
            "Middle panels summarize the capped first-year recovery distribution by core-area class. "
            "Right panels show the raw over-recovery share (>100%) together with the capped median recovery. "
            "Only landslides with t0 ATT <= -20% enter this recovery diagnostic."
        ),
        x=0,
        y=-0.13,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#222222"),
    )
    path = figures_dir / "Dashboard_07B_Area_Recovery_Joint_Summary.html"
    return [_write_html_with_fallback(fig, path)]


def plot_area_size_combined_summary(
    att_summary: pd.DataFrame,
    significance: pd.DataFrame,
    individual_recovery: pd.DataFrame,
    recovery_trend: pd.DataFrame,
    bin_summary: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """把 07A 和 07B 的有效信息拼成一张图，去掉表现不稳定的密度面。"""
    figures_dir = ensure_dir(figures_dir)
    if att_summary.empty or individual_recovery.empty or bin_summary.empty:
        return []

    outcomes = _ordered_outcomes(att_summary["outcome"].dropna().unique().tolist())
    rows = len(outcomes)
    if rows == 0:
        return []
    fig = make_subplots(
        rows=rows,
        cols=4,
        horizontal_spacing=0.065,
        vertical_spacing=0.10,
        column_widths=[0.28, 0.27, 0.22, 0.23],
    )
    styles = {
        "area_weighted_all": dict(dash="solid", width=3.0),
        "no_inner_small": dict(dash="dash", width=2.0),
        "inner_available_large": dict(dash="dot", width=2.4),
    }
    symbols = {"no_inner_small": "circle-open", "inner_available_large": "circle"}
    _, labels = _fixed_area_bins_labels()
    letters = list("abcdefghijklmnopqrstuvwxyz")
    letter_idx = 0

    for row, outcome in enumerate(outcomes, start=1):
        color = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        odf = att_summary[att_summary["outcome"] == outcome].copy()

        # 07A 左列：大小滑坡与面积加权总体 ATT% 轨迹。
        for series in ["area_weighted_all", "no_inner_small", "inner_available_large"]:
            sdf = odf[odf["series"] == series].sort_values("relative_year")
            if sdf.empty:
                continue
            ci_low = pd.to_numeric(sdf["ci_low_percent"], errors="coerce")
            ci_high = pd.to_numeric(sdf["ci_high_percent"], errors="coerce")
            if series == "area_weighted_all" and ci_low.notna().any() and ci_high.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([sdf["relative_year"], sdf["relative_year"].iloc[::-1]], ignore_index=True),
                        y=pd.concat([ci_high, ci_low.iloc[::-1]], ignore_index=True),
                        mode="lines",
                        name="95% CI",
                        legendgroup="area_weighted_ci",
                        showlegend=False,
                        fill="toself",
                        fillcolor=_rgba(color, 0.09),
                        line=dict(color="rgba(255,255,255,0)", width=0),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=2,
                )
            fig.add_trace(
                go.Scatter(
                    x=sdf["relative_year"],
                    y=sdf["att_percent"],
                    mode="lines+markers",
                    name=_series_label(series),
                    legendgroup=f"series_{series}",
                    showlegend=(row == 1),
                    line=dict(color=color, shape="spline", smoothing=0.45, **styles[series]),
                    marker=dict(size=5, color=color),
                    customdata=np.stack([sdf["n_landslides"], sdf["total_area_m2"]], axis=-1),
                    hovertemplate=(
                        "t=%{x}<br>ATT=%{y:.2f}%<br>"
                        "n=%{customdata[0]}<br>area=%{customdata[1]:.0f} m2<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
            if series in ["no_inner_small", "inner_available_large"] and ci_low.notna().any() and ci_high.notna().any():
                rel_numeric = pd.to_numeric(sdf["relative_year"], errors="coerce")
                post_rels = sorted(rel_numeric[rel_numeric > 0].dropna().astype(int).unique().tolist())
                key_rels = [0]
                if post_rels:
                    key_rels.append(post_rels[0])
                    key_rels.append(post_rels[-1])
                key_rels = list(dict.fromkeys(key_rels))
                key_sdf = sdf[rel_numeric.isin(key_rels)].copy()
                if not key_sdf.empty:
                    key_low = pd.to_numeric(key_sdf["ci_low_percent"], errors="coerce")
                    key_high = pd.to_numeric(key_sdf["ci_high_percent"], errors="coerce")
                    fig.add_trace(
                        go.Scatter(
                            x=key_sdf["relative_year"],
                            y=key_sdf["att_percent"],
                            mode="markers",
                            name=f"{_series_label(series)} key 95% CI",
                            legendgroup=f"key_ci_{series}",
                            showlegend=False,
                            marker=dict(size=3.2, color=color, opacity=0.72),
                            error_y=dict(
                                type="data",
                                symmetric=False,
                                array=(key_high - key_sdf["att_percent"]).clip(lower=0),
                                arrayminus=(key_sdf["att_percent"] - key_low).clip(lower=0),
                                thickness=0.9,
                                width=3,
                                color=_rgba(color, 0.60),
                            ),
                            hovertemplate="key-year CI<br>t=%{x}<br>ATT=%{y:.2f}%<extra></extra>",
                        ),
                        row=row,
                        col=1,
                    )
        fig.add_hline(y=0, line=dict(color="black", width=1), row=row, col=1)
        fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"), row=row, col=1)
        sig0 = significance[(significance["outcome"] == outcome) & (significance["relative_year"] == 0)]
        if not sig0.empty and pd.notna(sig0.iloc[0].get("p_value")):
            p = float(sig0.iloc[0]["p_value"])
            diff = float(sig0.iloc[0]["large_minus_small_att_percent"])
            fig.add_annotation(
                text=f"large-small t0={diff:.1f} pp; p={p:.2g}",
                x=0.98,
                y=0.92,
                xref=_axis_domain_ref("x", (row - 1) * 4 + 1),
                yref=_axis_domain_ref("y", (row - 1) * 4 + 1),
                showarrow=False,
                align="right",
                font=dict(size=10, color="#333333"),
            )

        # 07A 右列：每个滑坡的面积-第一年恢复率散点，只保留散点、IQR 和分箱中位数。
        rdf = individual_recovery[individual_recovery["outcome"] == outcome].copy()
        valid_rdf = rdf[
            rdf.get("valid_recovery_scatter", pd.Series(False, index=rdf.index)).astype(bool)
        ].dropna(subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"]).copy()
        valid_rdf = valid_rdf[valid_rdf["landslide_core_area_m2"] > 0]
        if not valid_rdf.empty:
            for series in ["no_inner_small", "inner_available_large"]:
                sdf = valid_rdf[valid_rdf["size_class"] == series].copy()
                if sdf.empty:
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=sdf["landslide_core_area_m2"],
                        y=sdf["first_year_recovery_percent_capped"],
                        mode="markers",
                        name=_series_label(series),
                        legendgroup=f"points_{series}",
                        showlegend=False,
                        marker=dict(
                            size=4.5 if series == "no_inner_small" else 5.5,
                            color=color,
                            opacity=0.24 if series == "no_inner_small" else 0.40,
                            symbol=symbols.get(series, "circle"),
                            line=dict(color=color, width=0.8),
                        ),
                        customdata=np.stack(
                            [
                                sdf["landslide_core_area_m2"],
                                sdf["event_year"],
                                sdf["event_att_percent"],
                                sdf["first_year_recovery_percent_raw"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "area=%{customdata[0]:.0f} m2<br>first-year recovery=%{y:.2f}%<br>"
                            "event year=%{customdata[1]:.0f}<br>t0 ATT=%{customdata[2]:.2f}%<br>"
                            "raw first-year recovery=%{customdata[3]:.2f}%<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=2,
                )
        tdf = recovery_trend[recovery_trend["outcome"] == outcome].copy()
        if not tdf.empty:
            tdf = tdf[tdf["area_median_m2"] > 0].copy()
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([tdf["area_median_m2"], tdf["area_median_m2"].iloc[::-1]], ignore_index=True),
                    y=pd.concat(
                        [tdf["first_year_recovery_q75_percent"], tdf["first_year_recovery_q25_percent"].iloc[::-1]],
                        ignore_index=True,
                    ),
                    mode="lines",
                    name="IQR",
                    legendgroup="binned_iqr",
                    showlegend=False,
                    fill="toself",
                    fillcolor="rgba(80,80,80,0.16)",
                    line=dict(color="rgba(80,80,80,0)", width=0),
                    hoverinfo="skip",
                ),
                row=row,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=tdf["area_median_m2"],
                    y=tdf["first_year_recovery_median_percent"],
                    mode="lines+markers",
                    name="binned median",
                    legendgroup="binned_median",
                    showlegend=(row == 1),
                    line=dict(color="black", width=2.2, shape="spline", smoothing=0.35),
                    marker=dict(size=6, color="black"),
                    customdata=np.stack([tdf["n_events"], tdf["area_min_m2"], tdf["area_max_m2"]], axis=-1),
                    hovertemplate=(
                        "median area=%{x:.0f} m2<br>median recovery=%{y:.2f}%<br>"
                        "n=%{customdata[0]:.0f}<br>area range=%{customdata[1]:.0f}-%{customdata[2]:.0f} m2"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=2,
            )

        # 07B 中列：面积等级分位数摘要，替代传统箱型图，避免 0/100 边界把箱体拉满。
        if not valid_rdf.empty:
            valid_rdf = valid_rdf.copy()
            bins, labels = _fixed_area_bins_labels()
            valid_rdf["area_bin_label"] = pd.cut(
                valid_rdf["landslide_core_area_m2"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            for label in labels:
                bdf = valid_rdf[valid_rdf["area_bin_label"].astype(str) == label]
                if bdf.empty:
                    continue
                y = pd.to_numeric(bdf["first_year_recovery_percent_capped"], errors="coerce").dropna()
                if y.empty:
                    continue
                q05 = float(y.quantile(0.05))
                q25 = float(y.quantile(0.25))
                med = float(y.median())
                q75 = float(y.quantile(0.75))
                q95 = float(y.quantile(0.95))
                n_bin = int(len(y))
                fig.add_trace(
                    go.Box(
                        x=[label],
                        q1=[q25],
                        median=[med],
                        q3=[q75],
                        lowerfence=[q05],
                        upperfence=[q95],
                        name="percentile box",
                        legendgroup="percentile_box",
                        showlegend=(row == 1 and label == labels[0]),
                        boxpoints=False,
                        fillcolor=_rgba(color, 0.20),
                        line=dict(color=color, width=1.25),
                        marker=dict(color=color, size=3),
                        width=0.46,
                        customdata=[[n_bin, q05, q25, med, q75, q95]],
                        hovertemplate=(
                            f"area bin={label}<br>n=%{{customdata[0]}}<br>"
                            "p05=%{customdata[1]:.1f}%<br>p25=%{customdata[2]:.1f}%<br>"
                            "median=%{customdata[3]:.1f}%<br>p75=%{customdata[4]:.1f}%<br>"
                            "p95=%{customdata[5]:.1f}%<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=3,
                )

        # 07B 右列：over-recovery 比例和分箱恢复率中位数。
        sdf = bin_summary[bin_summary["outcome"] == outcome].copy()
        if not sdf.empty:
            fig.add_trace(
                go.Bar(
                    x=sdf["area_bin_label"],
                    y=sdf["over_recovery_share_percent"],
                    marker=dict(color=color, opacity=0.70),
                    name="over-recovery",
                    legendgroup="over_recovery",
                    showlegend=(row == 1),
                    customdata=np.stack([sdf["n_events"], sdf["recovery_median_percent"]], axis=-1),
                    hovertemplate=(
                        "area bin=%{x}<br>over-recovery=%{y:.1f}%<br>"
                        "n=%{customdata[0]}<br>median recovery=%{customdata[1]:.1f}%<extra></extra>"
                    ),
                ),
                row=row,
                col=4,
            )
            fig.add_trace(
                go.Scatter(
                    x=sdf["area_bin_label"],
                    y=sdf["recovery_median_percent"],
                    mode="lines+markers",
                    line=dict(color="black", width=2),
                    marker=dict(color="black", size=5),
                    name="median recovery",
                    legendgroup="median_recovery",
                    showlegend=(row == 1),
                    hovertemplate="area bin=%{x}<br>median recovery=%{y:.1f}%<extra></extra>",
                ),
                row=row,
                col=4,
            )

        for col in [1, 2, 3, 4]:
            fig.add_annotation(
                text=f"({letters[letter_idx]})",
                x=0.01,
                y=0.94,
                xref=_axis_domain_ref("x", (row - 1) * 4 + col),
                yref=_axis_domain_ref("y", (row - 1) * 4 + col),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
            letter_idx += 1

    area_ticks = [100, 300, 1000, 3000, 10000, 30000, 100000]
    fig.update_layout(
        title=dict(
            text="Area-size DID recovery summary",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial", color="#111111"),
        ),
        template="plotly_white",
        width=2100,
        height=max(500, 330 * rows + 150),
        margin=dict(l=85, r=190, t=78, b=150),
        legend=dict(x=1.01, y=0.96, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family="Arial", size=12, color="#111111"),
    )
    for row in range(1, rows + 1):
        fig.update_xaxes(title_text="Years since landslide" if row == rows else "", showline=True, mirror=True, ticks="outside", row=row, col=1)
        fig.update_yaxes(title_text="Area-weighted DID ATT (%)" if row == 2 or rows == 1 else "", showline=True, mirror=True, ticks="outside", row=row, col=1)
        fig.update_xaxes(
            title_text="Core area (m2, log scale)" if row == rows else "",
            type="log",
            tickmode="array",
            tickvals=area_ticks,
            ticktext=[str(x) for x in area_ticks],
            showline=True,
            mirror=True,
            ticks="outside",
            row=row,
            col=2,
        )
        fig.update_yaxes(title_text="First-year recovery (%)" if row == 2 or rows == 1 else "", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=2)
        fig.update_xaxes(title_text="Core area class (m2)" if row == rows else "", showline=True, mirror=True, ticks="outside", tickangle=35, row=row, col=3)
        fig.update_yaxes(title_text="", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=3)
        fig.update_xaxes(title_text="Core area class (m2)" if row == rows else "", showline=True, mirror=True, ticks="outside", tickangle=35, row=row, col=4)
        fig.update_yaxes(title_text="Share / median (%)" if row == 2 or rows == 1 else "", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=4)
    fig.add_annotation(
        text=(
            "Area-size DID recovery summary in town Jiangwan. "
            "(a,e,i) Area-weighted DID ATT% trajectories, where larger landslides contribute more weight to the cohort-level curve. "
            "(b,f,j) Individual-landslide core area versus first-year DID recovery; the black line is the binned median and the grey band is the IQR. "
            "(c,g,k) First-year recovery percentile summaries by core-area class, with boxes showing IQR and whiskers showing the 5-95% range. "
            "(d,h,l) Over-recovery share and capped median recovery by core-area class. "
            "The removed density surface is intentionally not shown here because discrete 10 m raster areas and 0/100% clipping created misleading edge hotspots."
        ),
        x=0,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#222222"),
    )
    path = figures_dir / "Dashboard_06_Area_Size_Combined_Summary.html"
    return [_write_html_with_fallback(fig, path)]


def build_area_size_att_tables(
    event_percent: pd.DataFrame,
    class_summary: pd.DataFrame,
    zone_name: str = "core",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    基于 DID ATT% 构建面积加权恢复拓展表。

    多个滑坡之间的总体口径：
    spatial area-weighted ATT% = sum(area_i * ATT%_i) / sum(area_i)

    这里直接使用已经按面积加权估计出的 small / large DID ATT%，
    再按两类总面积合成为 overall 曲线。
    """
    if event_percent.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required = {
        "outcome",
        "zone_name",
        "relative_year",
        "att_percent",
        "ci_low_percent",
        "ci_high_percent",
        "size_class",
        "size_zone_name",
        "total_area_m2",
    }
    missing = required - set(event_percent.columns)
    if missing:
        raise ValueError(f"Event-percent table missing columns for area-size ATT response: {sorted(missing)}")

    df = event_percent.copy()
    df = df[(df["zone_name"].astype(str) == str(zone_name)) & (df["size_zone_name"].astype(str) == str(zone_name))].copy()
    size_classes = _size_classes_in_order(df["size_class"])
    df = df[df["size_class"].astype(str).isin(size_classes)].copy()
    for col in ["relative_year", "att_percent", "ci_low_percent", "ci_high_percent", "total_area_m2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["relative_year", "att_percent", "total_area_m2"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    class_att = df.rename(columns={"size_class": "series"}).copy()
    class_att["se"] = class_att.apply(_se_from_ci, axis=1)
    if "n_treated" in class_att.columns:
        class_att["n_landslides"] = pd.to_numeric(class_att["n_treated"], errors="coerce")
    else:
        class_att["n_landslides"] = np.nan
    keep_cols = [
        "outcome",
        "zone_name",
        "relative_year",
        "series",
        "att_percent",
        "ci_low_percent",
        "ci_high_percent",
        "se",
        "n_landslides",
        "total_area_m2",
    ]
    class_att = class_att[keep_cols]
    all_att = _weighted_all_att_percent(class_att)
    att_summary = pd.concat([all_att, class_att], ignore_index=True, sort=False).sort_values(
        ["outcome", "series", "relative_year"]
    )

    sig_rows: list[dict[str, object]] = []
    for (outcome, rel), chunk in class_att.groupby(["outcome", "relative_year"], dropna=True):
        ordered = _size_classes_in_order(chunk["series"])
        if len(ordered) < 2:
            continue
        small = chunk[chunk["series"] == ordered[0]]
        large = chunk[chunk["series"] == ordered[1]]
        if small.empty or large.empty:
            continue
        small_row = small.iloc[0]
        large_row = large.iloc[0]
        z, p = _normal_p_from_diff(
            small_row["att_percent"],
            small_row["se"],
            large_row["att_percent"],
            large_row["se"],
        )
        sig_rows.append(
            {
                "outcome": outcome,
                "zone_name": zone_name,
                "relative_year": int(rel),
                "small_att_percent": small_row["att_percent"],
                "large_att_percent": large_row["att_percent"],
                "large_minus_small_att_percent": large_row["att_percent"] - small_row["att_percent"],
                "z": z,
                "p_value": p,
            }
        )
    significance = pd.DataFrame(sig_rows).sort_values(["outcome", "relative_year"])

    recovery_rows: list[dict[str, object]] = []
    area_lookup = _class_area_lookup(class_summary, zone_name=zone_name)
    for (outcome, series), chunk in att_summary[att_summary["series"] != "area_weighted_all"].groupby(
        ["outcome", "series"], dropna=True
    ):
        curve = chunk.sort_values("relative_year").copy()
        event_row = curve[curve["relative_year"] == 0]
        post = curve[curve["relative_year"] > 0]
        if event_row.empty or post.empty:
            continue
        event_att = float(event_row.iloc[0]["att_percent"])
        latest = post.iloc[-1]
        latest_att = float(latest["att_percent"])
        latest_year = int(latest["relative_year"])
        if event_att < 0:
            latest_recovery = (latest_att - event_att) / (0.0 - event_att) * 100.0
        else:
            latest_recovery = np.nan
        recovery_rows.append(
            {
                "outcome": outcome,
                "zone_name": zone_name,
                "series": series,
                **area_lookup.get(series, {}),
                "event_att_percent": event_att,
                "latest_relative_year": latest_year,
                "latest_att_percent": latest_att,
                "latest_recovery_percent": latest_recovery,
                "recovery_speed_percent_per_year": latest_recovery / latest_year if latest_year > 0 and pd.notna(latest_recovery) else np.nan,
                "years_to_50pct_recovery": _years_to_recovery_threshold(curve, event_att, 50),
                "years_to_80pct_recovery": _years_to_recovery_threshold(curve, event_att, 80),
                "years_to_95pct_recovery": _years_to_recovery_threshold(curve, event_att, 95),
            }
        )
    recovery = pd.DataFrame(recovery_rows).sort_values(["outcome", "series"])
    return att_summary, significance, recovery


def _class_area_lookup(class_summary: pd.DataFrame, zone_name: str = "core") -> dict[str, dict[str, float]]:
    """读取 small/large 两类滑坡的代表面积。"""
    if class_summary.empty:
        return {}
    df = class_summary.copy()
    if "size_zone_name" in df.columns:
        df = df[df["size_zone_name"].astype(str) == str(zone_name)].copy()
    result: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        label = str(row.get("size_class"))
        result[label] = {
            "n_events": float(row.get("n_events", np.nan)),
            "total_core_area_m2": float(row.get("total_core_area_m2", np.nan)),
            "mean_core_area_m2": float(row.get("mean_core_area_m2", np.nan)),
            "median_core_area_m2": float(row.get("median_core_area_m2", np.nan)),
            "min_core_area_m2": float(row.get("min_core_area_m2", np.nan)),
            "max_core_area_m2": float(row.get("max_core_area_m2", np.nan)),
        }
    return result


def _years_to_recovery_threshold(curve: pd.DataFrame, event_att: float, threshold: float) -> float:
    """估算 ATT% 曲线恢复到灾时冲击的某个比例所需年份。"""
    if event_att >= 0:
        return np.nan
    post = curve[curve["relative_year"] >= 0].sort_values("relative_year").copy()
    if post.empty:
        return np.nan
    post["recovery_percent"] = (post["att_percent"] - event_att) / (0.0 - event_att) * 100.0
    hit = post[post["recovery_percent"] >= float(threshold)]
    if hit.empty:
        return np.nan
    first = hit.iloc[0]
    idx = post.index.get_loc(first.name)
    if idx == 0:
        return float(first["relative_year"])
    prev = post.iloc[idx - 1]
    x0, x1 = float(prev["relative_year"]), float(first["relative_year"])
    y0, y1 = float(prev["recovery_percent"]), float(first["recovery_percent"])
    if y1 == y0:
        return x1
    return x0 + (float(threshold) - y0) * (x1 - x0) / (y1 - y0)


def plot_landslide_effects_event_study(
    size_event_percent: pd.DataFrame,
    zone_event_percent: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig.5：大小滑坡和中心-外围的 DID 事件研究百分比曲线。"""
    figures_dir = ensure_dir(figures_dir)
    if size_event_percent.empty and zone_event_percent.empty:
        return []
    size_df = size_event_percent.copy()
    zone_df = zone_event_percent.copy()
    if not size_df.empty:
        size_df = size_df[
            (size_df.get("zone_name", "").astype(str) == "core")
            & (size_df.get("size_zone_name", "core").astype(str) == "core")
        ].copy()
    if not zone_df.empty:
        zone_df = zone_df[zone_df.get("zone_name", "").astype(str).isin(["ring_0_10m", "ring_10_20m"])].copy()
    outcomes = _ordered_outcomes(
        list(size_df.get("outcome", pd.Series(dtype=str)).dropna().unique())
        + list(zone_df.get("outcome", pd.Series(dtype=str)).dropna().unique())
    )
    if not outcomes:
        return []

    fig = make_subplots(rows=2, cols=len(outcomes), horizontal_spacing=0.08, vertical_spacing=0.16)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    zone_styles = {
        "ring_0_10m": ("0-10 m", "dash", 2.4, 0.25),
        "ring_10_20m": ("10-20 m", "dot", 2.4, 0.45),
    }
    size_styles = {
        "small": ("small (<1000 m2)", "dash", 2.3, 0.35),
        "large": ("large (>=1000 m2)", "solid", 2.8, 0.0),
    }
    for col, outcome in enumerate(outcomes, start=1):
        base = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        # 第一行：小滑坡 / 大滑坡。
        odf = size_df[size_df.get("outcome", "").astype(str) == outcome].copy() if not size_df.empty else pd.DataFrame()
        for series in _size_classes_in_order(odf.get("size_class", pd.Series(dtype=str))):
            sdf = odf[odf["size_class"].astype(str) == series].copy().sort_values("relative_year")
            sdf = sdf[sdf["relative_year"].isin(EVENT_STUDY_PLOT_RELATIVE_YEARS)]
            if sdf.empty:
                continue
            label, dash, width, mix = size_styles.get(series, (series, "solid", 2.2, 0.2))
            x, y = _smooth_xy(sdf["relative_year"], sdf["att_percent"])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=_mix_hex(base, "#ffffff", mix), width=width, dash=dash),
                    name=label,
                    legendgroup=f"size_{label}",
                    showlegend=(col == 1),
                    hovertemplate=f"{outcome} {label}<br>relative year=%{{x}}<br>ATT=%{{y:.1f}}%<extra></extra>",
                ),
                row=1,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=sdf["relative_year"],
                    y=sdf["att_percent"],
                    mode="markers",
                    marker=dict(color=_mix_hex(base, "#ffffff", mix), size=5.5),
                    showlegend=False,
                    hovertemplate=f"{outcome} {label}<br>relative year=%{{x}}<br>ATT=%{{y:.1f}}%<extra></extra>",
                ),
                row=1,
                col=col,
            )
        # 第二行：core / 外围环。
        zdf = zone_df[zone_df.get("outcome", "").astype(str) == outcome].copy() if not zone_df.empty else pd.DataFrame()
        for zone, (label, dash, width, mix) in zone_styles.items():
            sdf = zdf[zdf["zone_name"].astype(str) == zone].copy().sort_values("relative_year")
            sdf = sdf[sdf["relative_year"].isin(EVENT_STUDY_PLOT_RELATIVE_YEARS)]
            if sdf.empty:
                continue
            x, y = _smooth_xy(sdf["relative_year"], sdf["att_percent"])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color=_mix_hex(base, "#ffffff", mix), width=width, dash=dash),
                    name=label,
                    legendgroup=f"zone_{label}",
                    showlegend=(col == 1),
                    hovertemplate=f"{outcome} {label}<br>relative year=%{{x}}<br>ATT=%{{y:.1f}}%<extra></extra>",
                ),
                row=2,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=sdf["relative_year"],
                    y=sdf["att_percent"],
                    mode="markers",
                    marker=dict(color=_mix_hex(base, "#ffffff", mix), size=5.5),
                    showlegend=False,
                    hovertemplate=f"{outcome} {label}<br>relative year=%{{x}}<br>ATT=%{{y:.1f}}%<extra></extra>",
                ),
                row=2,
                col=col,
            )
        fig.add_annotation(
            text=outcome,
            x=0.5,
            y=1.08,
            xref=_axis_domain_ref("x", col),
            yref=_axis_domain_ref("y", col),
            showarrow=False,
            font=dict(size=14, color="#111111"),
        )

    subplot_letters = {}
    for col in range(1, len(outcomes) + 1):
        subplot_letters[(1, col)] = letters[(col - 1) * 2]
        subplot_letters[(2, col)] = letters[(col - 1) * 2 + 1]
    for row in [1, 2]:
        for col in range(1, len(outcomes) + 1):
            subplot_index = (row - 1) * len(outcomes) + col
            letter = subplot_letters[(row, col)]
            fig.add_annotation(
                text=f"({letter})",
                x=0.02,
                y=0.94,
                xref=_axis_domain_ref("x", subplot_index),
                yref=_axis_domain_ref("y", subplot_index),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
    fig.update_xaxes(
        range=[min(EVENT_STUDY_PLOT_RELATIVE_YEARS), max(EVENT_STUDY_PLOT_RELATIVE_YEARS)],
        tickmode="array",
        tickvals=EVENT_STUDY_PLOT_RELATIVE_YEARS,
        showline=True,
        mirror=True,
        ticks="outside",
    )
    fig.update_yaxes(
        range=EVENT_STUDY_PERCENT_RANGE,
        tickmode="array",
        tickvals=EVENT_STUDY_PERCENT_TICKS,
        showline=True,
        mirror=True,
        ticks="outside",
        zeroline=True,
        zerolinecolor="black",
    )
    fig.update_yaxes(title_text="DID ATT (%)", row=1, col=1)
    fig.update_yaxes(title_text="DID ATT (%)", row=2, col=1)
    for col in range(1, len(outcomes) + 1):
        fig.update_xaxes(title_text="Relative year", row=2, col=col)
    fig.update_layout(
        title_text="Fig. 5 Effects of landslides on vegetation cover and productivity",
        template="plotly_white",
        width=max(980, 360 * len(outcomes) + 180),
        height=680,
        margin=dict(l=80, r=190, t=90, b=115),
        legend=dict(x=1.01, y=0.5, xanchor="left", yanchor="middle"),
        font=dict(family="Arial", size=12, color="#111111"),
    )
    _add_caption(
        fig,
        "Fig. 5. Effects of landslides on vegetation cover and productivity in Jiangwan. "
        "(a, c, e) Core-area event-study ATT% split by small (<1000 m2) and large (>=1000 m2) landslides. "
        "(b, d, f) Periphery event-study ATT% for 0-10 m and 10-20 m zones.",
    )
    path = figures_dir / "Dashboard_05_Effects_Event_Study.html"
    return [_write_html_with_fallback(fig, path)]


def plot_area_size_att_response(
    att_summary: pd.DataFrame,
    significance: pd.DataFrame,
    recovery: pd.DataFrame,
    individual_recovery: pd.DataFrame,
    recovery_trend: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """绘制大小滑坡 DID ATT% 恢复拓展图。"""
    figures_dir = ensure_dir(figures_dir)
    if att_summary.empty:
        return []

    outcomes = _ordered_outcomes(att_summary["outcome"].dropna().unique().tolist())
    rows = len(outcomes)
    fig = make_subplots(rows=rows, cols=2, horizontal_spacing=0.12, vertical_spacing=0.10)
    styles = {
        "area_weighted_all": dict(dash="solid", width=3.0),
        "no_inner_small": dict(dash="dash", width=2.0),
        "inner_available_large": dict(dash="dot", width=2.4),
    }
    symbols = {"no_inner_small": "circle-open", "inner_available_large": "circle"}
    letters = list("abcdefghijklmnopqrstuvwxyz")
    letter_idx = 0

    for row, outcome in enumerate(outcomes, start=1):
        color = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        odf = att_summary[att_summary["outcome"] == outcome].copy()
        for series in ["area_weighted_all", "no_inner_small", "inner_available_large"]:
            sdf = odf[odf["series"] == series].sort_values("relative_year")
            if sdf.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sdf["relative_year"],
                    y=sdf["att_percent"],
                    mode="lines+markers",
                    name=_series_label(series),
                    legendgroup=series,
                    showlegend=(row == 1),
                    line=dict(color=color, shape="spline", smoothing=0.45, **styles[series]),
                    marker=dict(size=5, color=color),
                    customdata=np.stack([sdf["n_landslides"], sdf["total_area_m2"]], axis=-1),
                    hovertemplate="t=%{x}<br>ATT=%{y:.2f}%<br>n=%{customdata[0]}<br>area=%{customdata[1]:.0f} m2<extra></extra>",
                ),
                row=row,
                col=1,
            )

        fig.add_hline(y=0, line=dict(color="black", width=1), row=row, col=1)
        fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"), row=row, col=1)
        sig0 = significance[(significance["outcome"] == outcome) & (significance["relative_year"] == 0)]
        if not sig0.empty and pd.notna(sig0.iloc[0].get("p_value")):
            p = float(sig0.iloc[0]["p_value"])
            diff = float(sig0.iloc[0]["large_minus_small_att_percent"])
            fig.add_annotation(
                text=f"large-small t0={diff:.1f} pp; p={p:.2g}",
                x=0.98,
                y=0.92,
                xref=_axis_domain_ref("x", (row - 1) * 2 + 1),
                yref=_axis_domain_ref("y", (row - 1) * 2 + 1),
                showarrow=False,
                align="right",
                font=dict(size=10, color="#333333"),
            )

        rdf = individual_recovery[individual_recovery["outcome"] == outcome].copy()
        valid_rdf = rdf[
            rdf.get("valid_recovery_scatter", pd.Series(False, index=rdf.index)).astype(bool)
        ].dropna(subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"]).copy()
        valid_rdf = valid_rdf[valid_rdf["landslide_core_area_m2"] > 0]
        if not valid_rdf.empty:
            valid_rdf["log10_area"] = np.log10(valid_rdf["landslide_core_area_m2"].astype(float))
            fig.add_trace(
                go.Histogram2dContour(
                    x=valid_rdf["log10_area"],
                    y=valid_rdf["first_year_recovery_percent_capped"],
                    name="density",
                    legendgroup=f"density_{outcome}",
                    showlegend=False,
                    colorscale=OUTCOME_DENSITY_COLORS.get(outcome, OUTCOME_DENSITY_COLORS["LAI_SR2"]),
                    contours=dict(coloring="heatmap", showlines=True),
                    line=dict(color=color, width=0.6),
                    ncontours=9,
                    nbinsx=26,
                    nbinsy=20,
                    opacity=0.72,
                    hovertemplate="log10(area)=%{x:.2f}<br>recovery=%{y:.1f}%<br>density count=%{z}<extra></extra>",
                    colorbar=None,
                ),
                row=row,
                col=2,
            )
        for series in ["no_inner_small", "inner_available_large"]:
            sdf = rdf[(rdf["size_class"] == series) & (rdf["valid_recovery_scatter"].astype(bool))].dropna(
                subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"]
            )
            if sdf.empty:
                continue
            sdf = sdf[sdf["landslide_core_area_m2"] > 0].copy()
            sdf["log10_area"] = np.log10(sdf["landslide_core_area_m2"].astype(float))
            fig.add_trace(
                go.Scattergl(
                    x=sdf["log10_area"],
                    y=sdf["first_year_recovery_percent_capped"],
                    mode="markers",
                    name=_series_label(series),
                    legendgroup=f"point_{series}",
                    showlegend=False,
                    marker=dict(
                        size=5 if series == "no_inner_small" else 6,
                        color=color,
                        opacity=0.28 if series == "no_inner_small" else 0.42,
                        symbol=symbols.get(series, "circle"),
                        line=dict(color=color, width=0.8),
                    ),
                    customdata=np.stack(
                        [
                            sdf["landslide_core_area_m2"],
                            sdf["event_year"],
                            sdf["event_att_percent"],
                            sdf["first_year_recovery_percent_raw"],
                            sdf["latest_recovery_percent_raw"],
                            sdf["years_to_80pct_recovery"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "area=%{customdata[0]:.0f} m2<br>first-year recovery=%{y:.2f}%<br>"
                        "event year=%{customdata[1]:.0f}<br>t0 ATT=%{customdata[2]:.2f}%<br>"
                        "raw first-year recovery=%{customdata[3]:.2f}%<br>"
                        "raw latest recovery=%{customdata[4]:.2f}%<br>years to 80%=%{customdata[5]:.2f}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=2,
            )
        tdf = recovery_trend[recovery_trend["outcome"] == outcome].copy()
        if not tdf.empty:
            tdf = tdf[tdf["area_median_m2"] > 0].copy()
            tdf["log10_area"] = np.log10(tdf["area_median_m2"].astype(float))
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([tdf["log10_area"], tdf["log10_area"].iloc[::-1]], ignore_index=True),
                    y=pd.concat(
                        [tdf["first_year_recovery_q75_percent"], tdf["first_year_recovery_q25_percent"].iloc[::-1]],
                        ignore_index=True,
                    ),
                    mode="lines",
                    name="IQR",
                    legendgroup="binned_iqr",
                    showlegend=False,
                    fill="toself",
                    fillcolor="rgba(80,80,80,0.16)",
                    line=dict(color="rgba(80,80,80,0)", width=0),
                    hoverinfo="skip",
                ),
                row=row,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=tdf["log10_area"],
                    y=tdf["first_year_recovery_median_percent"],
                    mode="lines+markers",
                    name="binned median",
                    legendgroup="binned_median",
                    showlegend=(row == 1),
                    line=dict(color="black", width=2.4, shape="spline", smoothing=0.35),
                    marker=dict(size=7, color="black"),
                    customdata=np.stack([tdf["area_median_m2"], tdf["n_events"], tdf["area_min_m2"], tdf["area_max_m2"]], axis=-1),
                    hovertemplate=(
                        "median area=%{customdata[0]:.0f} m2<br>median first-year recovery=%{y:.2f}%<br>"
                        "n=%{customdata[1]:.0f}<br>area range=%{customdata[2]:.0f}-%{customdata[3]:.0f} m2"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=2,
            )
        area_ticks = [100, 300, 1000, 3000, 10000, 30000, 100000]
        fig.update_xaxes(
            tickmode="array",
            tickvals=[np.log10(x) for x in area_ticks],
            ticktext=[str(x) for x in area_ticks],
            row=row,
            col=2,
        )

        for col in [1, 2]:
            fig.add_annotation(
                text=f"({letters[letter_idx]})",
                x=0.01,
                y=0.94,
                xref=_axis_domain_ref("x", (row - 1) * 2 + col),
                yref=_axis_domain_ref("y", (row - 1) * 2 + col),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
            letter_idx += 1

    fig.update_layout(
        title=dict(
            text="Area-size extension of event study",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial", color="#111111"),
        ),
        template="plotly_white",
        width=1500,
        height=max(420, 320 * rows + 145),
        margin=dict(l=90, r=180, t=78, b=150),
        legend=dict(x=1.02, y=0.95, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0)"),
        font=dict(family="Arial", size=13, color="#111111"),
    )
    for row in range(1, rows + 1):
        fig.update_xaxes(title_text="Years since landslide" if row == rows else "", showline=True, mirror=True, ticks="outside", row=row, col=1)
        fig.update_yaxes(title_text="Area-weighted DID ATT (%)" if row == 2 or rows == 1 else "", showline=True, mirror=True, ticks="outside", row=row, col=1)
        fig.update_xaxes(title_text="Core area (m2, log scale)" if row == rows else "", showline=True, mirror=True, ticks="outside", row=row, col=2)
        fig.update_yaxes(
            title_text="First-year DID recovery (%)" if row == 2 or rows == 1 else "",
            range=[0, 100],
            showline=True,
            mirror=True,
            ticks="outside",
            row=row,
            col=2,
        )
    fig.add_annotation(
        text=(
            "Area-size extension of event study in town Jiangwan. "
            "(a,c,e) Area-weighted DID ATT% trajectories for all, small/no-inner-core and large/inner-core landslides. "
            "(b,d,f) Individual-landslide core area versus first-year DID recovery; the density layer is exploratory, while the black line and grey band show binned median and IQR. "
            "Only landslides with t0 ATT <= -20% are shown, and recovery is capped to 0-100% for visualization."
        ),
        x=0,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#222222"),
    )
    path = figures_dir / "Dashboard_07_Area_Size_Response.html"
    return [_write_html_with_fallback(fig, path)]


def plot_area_size_combined_summary_v2(
    att_summary: pd.DataFrame,
    individual_recovery: pd.DataFrame,
    bin_summary: pd.DataFrame,
    event_study_percent: pd.DataFrame,
    figures_dir: str | Path,
    drop_metrics: pd.DataFrame | None = None,
) -> list[Path]:
    """绘制精简版面积-恢复主图：大小滑坡、总体 Event Study 与面积等级恢复摘要。"""
    figures_dir = ensure_dir(figures_dir)
    if att_summary.empty or individual_recovery.empty:
        return []
    recomputed_bin_summary = _fixed_area_bin_summary(individual_recovery)
    if not recomputed_bin_summary.empty:
        bin_summary = recomputed_bin_summary
    if bin_summary.empty:
        return []

    att_summary = att_summary.copy()
    event_study_percent = event_study_percent.copy()
    drop_metrics = drop_metrics.copy() if drop_metrics is not None else pd.DataFrame()
    for display_df in (att_summary, event_study_percent):
        display_df["relative_year"] = pd.to_numeric(display_df["relative_year"], errors="coerce")
        display_df.dropna(subset=["relative_year"], inplace=True)
        display_df["relative_year"] = display_df["relative_year"].astype(int)
        display_df.drop(
            display_df.index[~display_df["relative_year"].isin(EVENT_STUDY_PLOT_RELATIVE_YEARS)],
            inplace=True,
        )
        if "att_percent" in display_df.columns:
            # 保持 DID ATT% 原始方向：生态下降为负值。
            display_df["plot_att_percent"] = pd.to_numeric(display_df["att_percent"], errors="coerce")
            display_df["plot_att_percent"] = pd.to_numeric(display_df["att_percent"], errors="coerce")
        if {"ci_low_percent", "ci_high_percent"}.issubset(display_df.columns):
            display_df["plot_ci_low_percent"] = pd.to_numeric(display_df["ci_low_percent"], errors="coerce")
            display_df["plot_ci_high_percent"] = pd.to_numeric(display_df["ci_high_percent"], errors="coerce")

    outcomes = _ordered_outcomes(att_summary["outcome"].dropna().unique().tolist())
    if not outcomes:
        return []
    rows = len(outcomes)
    fig = make_subplots(
        rows=rows,
        cols=3,
        horizontal_spacing=0.075,
        vertical_spacing=0.10,
        column_widths=[0.30, 0.34, 0.36],
    )
    letters = list("abcdefghijklmnopqrstuvwxyz")
    size_styles = {
        "no_inner_small": dict(dash="dash", width=2.3, label="small / no inner core"),
        "inner_available_large": dict(dash="solid", width=2.8, label="large / inner core"),
    }
    zone_styles = {
        "ring_0_10m": dict(dash="longdash", width=2.3, shade=0.38, label="0-10 m"),
        "ring_10_20m": dict(dash="dot", width=2.3, shade=0.62, label="10-20 m"),
    }
    drop_zone_specs = [
        ("no_inner_core", "no inner core"),
        ("core_inner", "inner core"),
        ("ring_0_10m", "0-10 m"),
        ("ring_10_20m", "10-20 m"),
    ]
    att_percent_range = EVENT_STUDY_PERCENT_RANGE
    att_percent_ticks = EVENT_STUDY_PERCENT_TICKS
    bins, labels = _fixed_area_bins_labels()
    drop_x_min, drop_x_max = -100.0, 0.0
    drop_bins = np.arange(drop_x_min, drop_x_max + 10.0, 10.0)
    drop_bin_centers = (drop_bins[:-1] + drop_bins[1:]) / 2.0
    if not drop_metrics.empty and {"outcome", "zone_name", "drop_percent"}.issubset(drop_metrics.columns):
        drop_metrics["drop_percent"] = pd.to_numeric(drop_metrics["drop_percent"], errors="coerce")
        drop_metrics = drop_metrics.dropna(subset=["drop_percent"]).copy()
        drop_metrics["drop_percent"] = drop_metrics["drop_percent"].clip(lower=0, upper=100)
        drop_metrics["change_percent"] = -drop_metrics["drop_percent"]
    else:
        drop_metrics = pd.DataFrame()

    def add_smooth_line(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        row: int,
        col: int,
        color: str,
        name: str,
        legendgroup: str,
        showlegend: bool,
        dash: str = "solid",
        width: float = 2.4,
        marker_size: float = 5.5,
        hovertemplate: str | None = None,
        low_col: str | None = None,
        high_col: str | None = None,
        error_alpha: float = 0.45,
    ) -> None:
        plot_df = data.dropna(subset=[x_col, y_col]).sort_values(x_col).copy()
        if plot_df.empty:
            return
        smooth_x, smooth_y = _smooth_xy(plot_df[x_col], plot_df[y_col])
        error_y = None
        if low_col and high_col and {low_col, high_col}.issubset(plot_df.columns):
            y = pd.to_numeric(plot_df[y_col], errors="coerce")
            low = pd.to_numeric(plot_df[low_col], errors="coerce")
            high = pd.to_numeric(plot_df[high_col], errors="coerce")
            upper = (high - y).clip(lower=0)
            lower = (y - low).clip(lower=0)
            if upper.notna().any() and lower.notna().any():
                error_y = dict(
                    type="data",
                    symmetric=False,
                    array=upper.fillna(0).to_numpy(),
                    arrayminus=lower.fillna(0).to_numpy(),
                    color=_rgba(color, error_alpha),
                    thickness=1.0,
                    width=3,
                )
        fig.add_trace(
            go.Scatter(
                x=smooth_x,
                y=smooth_y,
                mode="lines",
                line=dict(color=color, dash=dash, width=width),
                name=name,
                legendgroup=legendgroup,
                showlegend=showlegend,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_col],
                y=plot_df[y_col],
                mode="markers",
                marker=dict(size=marker_size, color=color, line=dict(color="white", width=0.55)),
                error_y=error_y,
                name=name,
                legendgroup=legendgroup,
                showlegend=False,
                customdata=plot_df,
                hovertemplate=hovertemplate,
                cliponaxis=False,
            ),
            row=row,
            col=col,
        )

    for r, outcome in enumerate(outcomes, start=1):
        base_color = OUTCOME_COLORS.get(outcome, "#4d4d4d")

        # 第二列：灾时下降组成矩阵；每行使用同一个 outcome 的全部滑坡事件作分母。
        ddf = drop_metrics[drop_metrics.get("outcome", pd.Series(dtype=str)).astype(str) == outcome].copy()
        if not ddf.empty:
            all_events = pd.Index(ddf["event_id"].dropna().astype(str).unique())
            denominator = max(float(len(all_events)), 1.0)
            inner_events = pd.Index(
                ddf.loc[ddf["zone_name"].astype(str) == "core_inner", "event_id"].dropna().astype(str).unique()
            )
            share_rows: list[np.ndarray] = []
            count_rows: list[np.ndarray] = []
            row_labels: list[str] = []
            for zone_name, label in drop_zone_specs:
                if zone_name == "no_inner_core":
                    no_inner_events = all_events.difference(inner_events)
                    zdf = ddf[
                        (ddf["zone_name"].astype(str) == "core")
                        & (ddf["event_id"].astype(str).isin(no_inner_events))
                    ].copy()
                else:
                    zdf = ddf[ddf["zone_name"].astype(str) == zone_name].copy()
                event_changes = (
                    zdf.dropna(subset=["event_id"])
                    .assign(event_id=lambda x: x["event_id"].astype(str))
                    .groupby("event_id", as_index=False)["change_percent"]
                    .mean()
                )
                missing_events = all_events.difference(pd.Index(event_changes["event_id"].astype(str)))
                if len(missing_events) > 0:
                    event_changes = pd.concat(
                        [
                            event_changes,
                            pd.DataFrame({"event_id": missing_events.astype(str), "change_percent": 0.0}),
                        ],
                        ignore_index=True,
                    )
                event_changes = event_changes[
                    (event_changes["change_percent"] >= drop_x_min)
                    & (event_changes["change_percent"] <= drop_x_max)
                ]
                counts, _ = np.histogram(event_changes["change_percent"], bins=drop_bins)
                counts = counts.astype(float)
                share_rows.append(counts / denominator * 100.0)
                count_rows.append(counts)
                row_labels.append(label)
            if share_rows:
                x_labels = [f"{int(drop_bins[i])} to {int(drop_bins[i + 1])}" for i in range(len(drop_bins) - 1)]
                share_matrix = np.vstack(share_rows)
                count_matrix = np.vstack(count_rows)
                fig.add_trace(
                    go.Heatmap(
                        x=x_labels,
                        y=row_labels,
                        z=share_matrix,
                        customdata=count_matrix,
                        colorscale=[
                            [0.0, "#ffffff"],
                            [0.35, _rgba(base_color, 0.28)],
                            [1.0, base_color],
                        ],
                        zmin=0,
                        zmax=60,
                        colorbar=dict(
                            title=dict(text="Share (%)", side="right"),
                            len=0.22,
                            thickness=12,
                            x=1.005,
                            y=0.5,
                        ) if r == 1 else None,
                        showscale=(r == 1),
                        hovertemplate=(
                            "Type=%{y}<br>"
                            "Change bin=%{x}%<br>"
                            "Share=%{z:.2f}%<br>"
                            "Count=%{customdata:.0f}<br>"
                            f"all events={int(denominator)}<extra></extra>"
                        ),
                        name=f"{outcome} drop composition",
                    ),
                    row=r,
                    col=2,
                )

        # 第一列：大小滑坡与两级外部环的 ATT% 轨迹。
        odf = att_summary[(att_summary["outcome"] == outcome) & (att_summary["series"].isin(size_styles))].copy()
        for series, style in size_styles.items():
            sdf = odf[odf["series"] == series].sort_values("relative_year")
            if sdf.empty:
                continue
            line_color = base_color if series == "inner_available_large" else _mix_hex(base_color, "#ffffff", 0.48)
            add_smooth_line(
                sdf,
                "relative_year",
                "plot_att_percent",
                r,
                1,
                line_color,
                style["label"],
                f"size_{series}",
                r == 1,
                dash=style["dash"],
                width=style["width"],
                hovertemplate="t=%{x}<br>area-weighted DID ATT=%{y:.2f}%<extra></extra>",
                low_col="plot_ci_low_percent",
                high_col="plot_ci_high_percent",
                error_alpha=0.34,
            )

        edf = event_study_percent[
            (event_study_percent.get("outcome", pd.Series(dtype=str)).astype(str) == outcome)
            & (event_study_percent.get("zone_name", pd.Series(dtype=str)).astype(str).isin(zone_styles))
        ].copy()
        for zone_name, style in zone_styles.items():
            zdf = edf[edf["zone_name"] == zone_name].sort_values("relative_year").copy()
            if zdf.empty:
                continue
            line_color = _mix_hex(base_color, "#ffffff", float(style["shade"]))
            add_smooth_line(
                zdf,
                "relative_year",
                "plot_att_percent",
                r,
                1,
                line_color,
                str(style["label"]),
                f"zone_{zone_name}",
                r == 1,
                dash=str(style["dash"]),
                width=float(style["width"]),
                hovertemplate="t=%{x}<br>DID ATT=%{y:.2f}%<extra></extra>",
                low_col="plot_ci_low_percent",
                high_col="plot_ci_high_percent",
                error_alpha=0.28,
            )
        fig.add_hline(y=0, line=dict(color="black", width=1), row=r, col=1)
        fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"), row=r, col=1)

        # 第三列：面积等级恢复分布 + 过恢复比例，合并旧箱型图和柱线图的信息。
        rdf = individual_recovery[
            (individual_recovery["outcome"] == outcome)
            & individual_recovery.get("valid_recovery_scatter", pd.Series(False, index=individual_recovery.index)).astype(bool)
        ].copy()
        rdf = rdf.dropna(subset=["landslide_core_area_m2", "first_year_recovery_percent_capped"])
        rdf = rdf[rdf["landslide_core_area_m2"] > 0].copy()
        if not rdf.empty:
            rdf["area_bin_label"] = pd.cut(
                rdf["landslide_core_area_m2"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            sdf = bin_summary[bin_summary["outcome"] == outcome].copy()
            box_labels: list[str] = []
            for label in labels:
                bdf = rdf[rdf["area_bin_label"].astype(str) == label]
                if bdf.empty:
                    continue
                y = pd.to_numeric(bdf["first_year_recovery_percent_capped"], errors="coerce").dropna()
                if len(y) < 5:
                    continue
                box_labels.append(str(label))
                fig.add_trace(
                    go.Box(
                        x=[label],
                        q1=[float(y.quantile(0.25))],
                        median=[float(y.median())],
                        q3=[float(y.quantile(0.75))],
                        lowerfence=[float(y.quantile(0.05))],
                        upperfence=[float(y.quantile(0.95))],
                        name="5-95% recovery box",
                        legendgroup="recovery_box",
                        showlegend=(r == 1 and label == labels[0]),
                        boxpoints=False,
                        fillcolor=_rgba(base_color, 0.16),
                        line=dict(color=base_color, width=1.2),
                        width=0.48,
                        hovertemplate=f"area={label}<br>median=%{{median:.1f}}%<extra></extra>",
                    ),
                    row=r,
                    col=3,
                )
            if not sdf.empty:
                sdf = sdf.sort_values("area_bin_order") if "area_bin_order" in sdf.columns else sdf
                if box_labels:
                    sdf = sdf[sdf["area_bin_label"].astype(str).isin(box_labels)].copy()
                if not sdf.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sdf["area_bin_label"],
                            y=sdf["over_recovery_share_percent"],
                            mode="lines+markers",
                            line=dict(color="black", width=2.0),
                            marker=dict(size=5.2, color="black"),
                            name="over-recovery share",
                            legendgroup="over_recovery",
                            showlegend=(r == 1),
                            hovertemplate="area=%{x}<br>over-recovery share=%{y:.1f}%<extra></extra>",
                        ),
                        row=r,
                        col=3,
                    )

        for c in range(1, 4):
            panel_index = (r - 1) * 3 + c
            fig.add_annotation(
                text=f"<b>({letters[panel_index - 1]})</b>",
                x=0.012,
                y=0.94,
                xref=_axis_domain_ref("x", panel_index),
                yref=_axis_domain_ref("y", panel_index),
                showarrow=False,
                font=dict(size=13, color="black"),
            )

    fig.update_layout(
        title=dict(
            text="Drop composition and area-size DID recovery summary",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial", color="#111111"),
        ),
        template="plotly_white",
        width=1720,
        height=max(760, 315 * rows + 150),
        margin=dict(l=86, r=230, t=78, b=165),
        legend=dict(x=1.01, y=0.96, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.86)"),
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=12, color="#111111"),
    )
    for row in range(1, rows + 1):
        fig.update_xaxes(
            title_text="Years since landslide" if row == rows else "",
            range=[-2, 5],
            showline=True,
            mirror=True,
            ticks="outside",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text="DID ATT (%)" if row == 2 or rows == 1 else "",
            range=att_percent_range,
            tickmode="array",
            tickvals=att_percent_ticks,
            showline=True,
            mirror=True,
            ticks="outside",
            row=row,
            col=1,
        )
        fig.update_xaxes(
            title_text="Change from t-1 to t0 (%)" if row == rows else "",
            showline=True,
            mirror=True,
            ticks="outside",
            tickangle=35,
            row=row,
            col=2,
        )
        fig.update_yaxes(
            title_text="Zone / type" if row == 2 or rows == 1 else "",
            showline=True,
            mirror=True,
            ticks="outside",
            row=row,
            col=2,
        )
        fig.update_xaxes(
            title_text="Core area class (m2)" if row == rows else "",
            showline=True,
            mirror=True,
            ticks="outside",
            tickangle=35,
            row=row,
            col=3,
        )
        fig.update_yaxes(title_text="First-year recovery / share (%)" if row == 2 or rows == 1 else "", range=[0, 100], showline=True, mirror=True, ticks="outside", row=row, col=3)
    _add_area_caption = (
        "Area-size DID recovery summary in town Jiangwan. "
        "(a,d,g) Event-study ATT trajectories for large/inner-core, small/no-inner-core, 0-10 m and 10-20 m series; error bars are derived from 95% CI and ecological decline is negative. "
        "(b,e,h) Event-year drop composition heatmaps; columns are t-1 to t0 percentage-change bins, rows are no-inner-core, inner-core, 0-10 m and 10-20 m types, and colors use all landslides of each outcome as the common denominator. "
        "(c,f,i) Core-area-class first-year recovery distributions; boxes show 5-95% ranges and IQR where n>=5, while black lines show over-recovery share (>100%), not the mean recovery."
    )
    fig.add_annotation(
        text=_add_area_caption,
        x=0,
        y=-0.13,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#222222"),
    )
    path = figures_dir / "Dashboard_06_Area_Size_Combined_Summary.html"
    return [_write_html_with_fallback(fig, path)]


def run_area_size_response_analysis(
    event_percent: pd.DataFrame,
    class_summary: pd.DataFrame,
    panel: pd.DataFrame,
    group_time: pd.DataFrame,
    assignment: pd.DataFrame,
    results_dir: str | Path,
    figures_dir: str | Path,
    event_study_percent: pd.DataFrame | None = None,
    zone_name: str = "core",
    value_column: str = "mean",
    reference_relative_year: int = -1,
    figure_modes: list[str] | tuple[str, ...] | None = None,
    drop_metrics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """生成基于 DID ATT% 的大小滑坡恢复拓展分析。"""
    results_dir = ensure_dir(results_dir)
    figure_modes = list(figure_modes or ["combined"])
    att_summary, significance, recovery = build_area_size_att_tables(
        event_percent=event_percent,
        class_summary=class_summary,
        zone_name=zone_name,
    )
    individual_att = _build_individual_att_percent(
        panel=panel,
        group_time=group_time,
        assignment=assignment,
        zone_name=zone_name,
        value_column=value_column,
        reference_relative_year=reference_relative_year,
    )
    # Fig.6/Fig.7 现在由 recovery_figures.py 统一按配置的反事实恢复阈值生成。
    # 这里仅保留大小滑坡 ATT% 和单滑坡 ATT% 基础表，避免继续写出旧的 recovery/k/80% 表。
    individual_recovery = pd.DataFrame()
    recovery_trend = pd.DataFrame()
    bin_summary = pd.DataFrame()
    safe_to_csv(att_summary, results_dir / "area_weighted_att_percent.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(significance, results_dir / "area_size_group_significance.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(individual_att, results_dir / "area_individual_att_percent.csv", index=False, encoding="utf-8-sig")
    if event_study_percent is None:
        event_percent_path = Path(results_dir) / "event_study_att_percent.csv"
        event_study_percent = pd.read_csv(event_percent_path) if event_percent_path.exists() else pd.DataFrame()
    if drop_metrics is None:
        drop_metrics_path = Path(results_dir) / "individual_zone_drop_metrics.csv"
        drop_metrics = pd.read_csv(drop_metrics_path) if drop_metrics_path.exists() else pd.DataFrame()
    if "basic" in figure_modes:
        plot_area_size_att_response(att_summary, significance, recovery, individual_recovery, recovery_trend, figures_dir)
    if "joint" in figure_modes:
        plot_area_recovery_joint_summary(individual_recovery, bin_summary, figures_dir)
    plot_landslide_effects_event_study(
        size_event_percent=event_percent,
        zone_event_percent=event_study_percent,
        figures_dir=figures_dir,
    )
    if "legacy_combined" in figure_modes:
        plot_area_size_combined_summary_v2(
            att_summary,
            individual_recovery,
            bin_summary,
            event_study_percent,
            figures_dir,
            drop_metrics=drop_metrics,
        )
    return {
        "att_summary": att_summary,
        "significance": significance,
        "recovery": recovery,
        "individual_att": individual_att,
        "individual_recovery": individual_recovery,
        "recovery_trend": recovery_trend,
        "bin_summary": bin_summary,
    }
