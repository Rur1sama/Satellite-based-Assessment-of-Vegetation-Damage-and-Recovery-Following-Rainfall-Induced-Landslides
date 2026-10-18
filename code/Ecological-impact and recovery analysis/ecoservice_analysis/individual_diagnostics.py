from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .io_utils import ensure_dir, safe_to_csv


OUTCOME_COLORS = {
    "LAI_SR2": "#b2182b",
    "FVC": "#1b7837",
    "FAPAR": "#e08214",
    "GPP": "#2166ac",
    "VOD": "#6a3d9a",
}

ZONE_ORDER = {
    "core": 0,
    "core_weighted": 1,
    "core_inner": 2,
    "ring_0_10m": 3,
    "ring_10_20m": 4,
}

DROP_CLASS_ORDER = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
DROP_CLASS_COLORS = {
    "0-20%": "#d9d9d9",
    "20-40%": "#fee08b",
    "40-60%": "#fdae61",
    "60-80%": "#f46d43",
    "80-100%": "#a50026",
}
DROP_CLASS_CODES = {label: idx + 1 for idx, label in enumerate(DROP_CLASS_ORDER)}
DROP_CLASS_RGB = {
    label: tuple(int(DROP_CLASS_COLORS[label].lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    for label in DROP_CLASS_ORDER
}


def _ordered_outcomes(values: list[str]) -> list[str]:
    """按照主图常用顺序排列 outcome。"""
    preferred = ["FVC", "LAI_SR2", "GPP"]
    allowed = set(preferred)
    rank = {name: idx for idx, name in enumerate(preferred)}
    return sorted([str(v) for v in values if str(v) in allowed], key=lambda x: (rank.get(x, 999), x))


def _ordered_zones(values: list[str]) -> list[str]:
    """按照 core -> core_inner -> 外部环的顺序排列 zone。"""
    return sorted([str(v) for v in values], key=lambda x: (ZONE_ORDER.get(x, 999), x))


def _drop_class(drop_percent: float) -> str:
    """把单个滑坡的百分比突降归入用户指定的 5 个等级。"""
    value = max(0.0, min(100.0, float(drop_percent)))
    if value >= 80:
        return "80-100%"
    if value >= 60:
        return "60-80%"
    if value >= 40:
        return "40-60%"
    if value >= 20:
        return "20-40%"
    return "0-20%"


def _zone_label(zone_name: str) -> str:
    """图上使用更容易读的 zone 名称。"""
    labels = {
        "core": "core",
        "core_weighted": "weighted core",
        "core_inner": "core inner",
        "ring_0_10m": "0-10 m",
        "ring_10_20m": "10-20 m",
    }
    return labels.get(str(zone_name), str(zone_name))


def _outcome_color(outcome: str) -> str:
    """获取 outcome 的主色。"""
    return OUTCOME_COLORS.get(str(outcome), "#4d4d4d")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """把 #RRGGBB 颜色转换成 Plotly 可用的 rgba 字符串。"""
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return f"rgba(80,80,80,{float(alpha)})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{float(alpha)})"


def _pchip_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """计算形状保持 Hermite 曲线斜率，避免样条曲线越过相邻样本点极值。"""
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


def _shape_preserving_curve(x_values: list[float], y_values: list[float], points_per_segment: int = 16) -> tuple[list[float], list[float]]:
    """生成经过原始节点且不制造额外极值的平滑曲线。"""
    curve = (
        pd.DataFrame({"x": pd.to_numeric(pd.Series(x_values), errors="coerce"), "y": pd.to_numeric(pd.Series(y_values), errors="coerce")})
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


def build_individual_drop_metrics(
    panel: pd.DataFrame,
    value_column: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 zone-outcome panel 中计算单个滑坡在发生当年的突降百分比。

    这里使用观测值口径：
    drop_percent = (t-1 的均值 - t0 的均值) / t-1 的均值 * 100。
    该结果用于“直观诊断”和空间分布，不替代 DID 的反事实平均效应。
    """
    if panel.empty or value_column not in panel.columns:
        empty = pd.DataFrame()
        return empty, empty, empty

    df = panel.copy()
    required = {
        "outcome",
        "unit_id",
        "zone_id",
        "event_id",
        "zone_name",
        "zone_role",
        "event_year",
        "relative_year",
        "area_m2",
        value_column,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing required columns for individual diagnostics: {sorted(missing)}")

    df = df[df["zone_role"] == "treated"].copy()
    df["relative_year"] = pd.to_numeric(df["relative_year"], errors="coerce")
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df[df["relative_year"].isin([-1, 0])].copy()
    df = df.dropna(subset=["event_year", value_column])
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    keys = [
        "outcome",
        "unit_id",
        "zone_id",
        "event_id",
        "zone_name",
        "event_year",
        "matched_event_year",
        "distance_min_m",
        "distance_max_m",
        "area_m2",
    ]
    stat_cols = [value_column, "std", "n_pixels", "median", "min", "max"]
    stat_cols = [col for col in stat_cols if col in df.columns]

    pre = df[df["relative_year"] == -1][keys + stat_cols].copy()
    post = df[df["relative_year"] == 0][keys + stat_cols].copy()
    rename_pre = {col: f"pre_{col}" for col in stat_cols}
    rename_post = {col: f"event_{col}" for col in stat_cols}
    pre = pre.rename(columns=rename_pre)
    post = post.rename(columns=rename_post)
    metrics = pre.merge(post, on=keys, how="inner")
    if metrics.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    pre_col = f"pre_{value_column}"
    event_col = f"event_{value_column}"
    metrics[pre_col] = pd.to_numeric(metrics[pre_col], errors="coerce")
    metrics[event_col] = pd.to_numeric(metrics[event_col], errors="coerce")
    metrics = metrics.dropna(subset=[pre_col, event_col])
    metrics = metrics[metrics[pre_col] > 0].copy()
    if metrics.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    metrics["delta"] = metrics[event_col] - metrics[pre_col]
    metrics["drop_percent_raw"] = (metrics[pre_col] - metrics[event_col]) / metrics[pre_col] * 100.0
    metrics["drop_percent"] = metrics["drop_percent_raw"].clip(lower=0.0, upper=100.0)
    metrics["drop_class"] = metrics["drop_percent_raw"].map(_drop_class)
    metrics["event_year"] = metrics["event_year"].astype(int)
    metrics["zone_label"] = metrics["zone_name"].map(_zone_label)

    summary = (
        metrics.groupby(["outcome", "zone_name", "event_year", "drop_class"], as_index=False)
        .agg(
            n_landslides=("event_id", "nunique"),
            total_area_m2=("area_m2", "sum"),
            mean_drop_percent=("drop_percent_raw", "mean"),
            sd_drop_percent=("drop_percent_raw", "std"),
            median_drop_percent=("drop_percent_raw", "median"),
            mean_pre_value=(pre_col, "mean"),
            mean_event_value=(event_col, "mean"),
        )
        .sort_values(["outcome", "zone_name", "event_year", "drop_class"])
    )

    value_summary = (
        panel[panel.get("zone_role", pd.Series(dtype=str)).eq("treated")]
        .copy()
    )
    if not value_summary.empty:
        value_summary["relative_year"] = pd.to_numeric(value_summary["relative_year"], errors="coerce")
        value_summary[value_column] = pd.to_numeric(value_summary[value_column], errors="coerce")
        value_summary = (
            value_summary.dropna(subset=[value_column])
            .groupby(["outcome", "zone_name", "relative_year"], as_index=False)
            .agg(
                n_zone_years=(value_column, "count"),
                mean_of_zone_means=(value_column, "mean"),
                sd_of_zone_means=(value_column, "std"),
                median_of_zone_means=(value_column, "median"),
                q05_of_zone_means=(value_column, lambda s: float(s.quantile(0.05))),
                q95_of_zone_means=(value_column, lambda s: float(s.quantile(0.95))),
                mean_within_zone_std=("std", "mean") if "std" in panel.columns else (value_column, "std"),
            )
            .sort_values(["outcome", "zone_name", "relative_year"])
        )
    return metrics, summary, value_summary


def _histogram_annotation(values: pd.Series) -> str:
    """生成直方图角标，显示样本量、均值和标准差。"""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "n=0"
    return f"n={len(values)}<br>mean={values.mean():.2f}<br>sd={values.std():.2f}"


def _add_caption(fig: go.Figure, text: str) -> None:
    """在 HTML 图底部添加图注。"""
    fig.add_annotation(
        text=text,
        x=0,
        y=-0.16,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        align="left",
        font=dict(size=12, color="#222"),
    )


def plot_value_histograms(panel: pd.DataFrame, figures_dir: str | Path, value_column: str = "mean") -> list[Path]:
    """绘制所有 zone-year 生态指标均值的分布直方图，用来查看平均值背后的样本形态。"""
    figures_dir = ensure_dir(figures_dir)
    df = panel[(panel["zone_role"] == "treated") & panel[value_column].notna()].copy()
    if df.empty:
        return []
    outcomes = _ordered_outcomes(df["outcome"].dropna().unique().tolist())
    zones = _ordered_zones(df["zone_name"].dropna().unique().tolist())
    fig = make_subplots(rows=len(outcomes), cols=len(zones), horizontal_spacing=0.055, vertical_spacing=0.10)

    for r, outcome in enumerate(outcomes, start=1):
        for c, zone in enumerate(zones, start=1):
            zdf = df[(df["outcome"] == outcome) & (df["zone_name"] == zone)].copy()
            if zdf.empty:
                continue
            color = _outcome_color(outcome)
            fig.add_trace(
                go.Histogram(
                    x=zdf[value_column],
                    nbinsx=40,
                    marker=dict(color=color, opacity=0.72, line=dict(color="white", width=0.4)),
                    name=f"{outcome} {_zone_label(zone)}",
                    showlegend=False,
                    hovertemplate="value=%{x:.4g}<br>count=%{y}<extra></extra>",
                ),
                row=r,
                col=c,
            )
            fig.add_annotation(
                text=f"({chr(96 + (r - 1) * len(zones) + c)}) {outcome} | {_zone_label(zone)}<br>{_histogram_annotation(zdf[value_column])}",
                x=0.03,
                y=0.95,
                xref=f"x{(r - 1) * len(zones) + c} domain" if (r - 1) * len(zones) + c > 1 else "x domain",
                yref=f"y{(r - 1) * len(zones) + c} domain" if (r - 1) * len(zones) + c > 1 else "y domain",
                showarrow=False,
                align="left",
                font=dict(size=10, color="#111"),
            )

    fig.update_layout(
        template="plotly_white",
        height=max(760, 280 * len(outcomes)),
        width=max(1180, 300 * len(zones)),
        bargap=0.03,
        margin=dict(l=70, r=30, t=78, b=130),
        title=dict(
            text="Value distribution histograms",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial, Microsoft YaHei, sans-serif", color="#111"),
        ),
    )
    fig.update_xaxes(showline=True, mirror=True, ticks="outside", title_text="")
    fig.update_yaxes(showline=True, mirror=True, ticks="outside", title_text="")
    fig.update_xaxes(title_text=value_column, row=len(outcomes), col=max(1, len(zones) // 2))
    fig.update_yaxes(title_text="Count", row=max(1, len(outcomes) // 2), col=1)
    _add_caption(
        fig,
        "Observed value distribution histograms in town Jiangwan. Each panel shows treated-zone annual means before DID aggregation; these plots are diagnostics for checking what the mean is averaging, and labels report n, mean and standard deviation.",
    )
    path = figures_dir / "Dashboard_03_Value_Distribution_Histograms.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return [path]


def plot_landslide_distribution_patterns(
    panel: pd.DataFrame,
    zones_path: str | Path,
    output_dir: str | Path,
    figures_dir: str | Path,
    value_column: str = "mean",
    study_area_path: str | Path | None = None,
) -> list[Path]:
    """Fig.4：按滑坡均值统计灾前 LAI 和坡度分布，并给出滑坡空间位置。"""
    figures_dir = ensure_dir(figures_dir)
    # Fig.4 由独立脚本负责构建“研究区背景宽柱 + 滑坡频率窄柱”的论文版布局。
    # 全流程已先运行该脚本时，直接复用成品，避免这里的旧诊断图覆盖新图。
    prepared_path = figures_dir / "Dashboard_04_Landslide_Distribution_Patterns.html"
    fig04_source_dir = Path(output_dir).parent / "06_figure_source_data" / "Fig04_Landslide_Distribution"
    required_overlay_tables = [
        fig04_source_dir / "Fig04B_PreEvent_LAI_OverlayFrequency.csv",
        fig04_source_dir / "Fig04C_Slope_OverlayFrequency.csv",
    ]
    if prepared_path.exists() and all(path.exists() for path in required_overlay_tables):
        return [prepared_path]
    if panel.empty:
        return []
    try:
        import geopandas as gpd
    except Exception:
        return []

    zones = gpd.read_file(zones_path)
    if zones.empty:
        return []
    core = zones[(zones.get("zone_role", "") == "treated") & (zones.get("zone_name", "") == "core")].copy()
    if core.empty:
        return []
    if "area_m2" not in core.columns:
        core["area_m2"] = core.geometry.area
    cent = core.geometry.centroid
    core["_x"] = cent.x
    core["_y"] = cent.y
    core["event_year"] = pd.to_numeric(core.get("event_year"), errors="coerce")
    core["area_m2"] = pd.to_numeric(core["area_m2"], errors="coerce")

    lai = panel[
        (panel.get("zone_role", "") == "treated")
        & (panel.get("zone_name", "") == "core")
        & (panel.get("outcome", "") == "LAI_SR2")
        & (pd.to_numeric(panel.get("relative_year"), errors="coerce") == -1)
    ].copy()
    lai[value_column] = pd.to_numeric(lai[value_column], errors="coerce")
    lai_event = (
        lai.dropna(subset=["event_id", value_column])
        .groupby("event_id", as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "pre_lai_mean"})
    )

    feature_path = Path(output_dir).parent / "06_figure_source_data" / "Fig04_Landslide_Distribution" / "Fig04_event_level_values.csv"
    if not feature_path.exists():
        feature_path = Path(output_dir) / "recovery_rate_feature_table.csv"
    slope_event = pd.DataFrame()
    if feature_path.exists():
        feature = pd.read_csv(feature_path)
        if {"event_id", "slope_mean"}.issubset(feature.columns):
            slope_event = feature[["event_id", "slope_mean"]].drop_duplicates("event_id").copy()
            slope_event["slope_mean"] = pd.to_numeric(slope_event["slope_mean"], errors="coerce")

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "histogram"}]],
        column_widths=[0.50, 0.25, 0.25],
        horizontal_spacing=0.08,
    )

    if study_area_path is not None and Path(study_area_path).exists():
        try:
            study = gpd.read_file(study_area_path).to_crs(core.crs)
            for geom in study.geometry:
                if geom is None or geom.is_empty:
                    continue
                geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
                for poly in geoms:
                    x, y = poly.exterior.xy
                    fig.add_trace(
                        go.Scatter(
                            x=list(x),
                            y=list(y),
                            mode="lines",
                            line=dict(color="#555555", width=1),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )
        except Exception:
            pass
    marker_size = np.sqrt(core["area_m2"].fillna(core["area_m2"].median()).clip(lower=1)) / 8.0
    marker_size = np.clip(marker_size, 3, 18)
    fig.add_trace(
        go.Scattergl(
            x=core["_x"],
            y=core["_y"],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=core["event_year"],
                colorscale="Viridis",
                colorbar=dict(title="Event year", len=0.70, y=0.50),
                opacity=0.72,
                line=dict(color="rgba(0,0,0,0.35)", width=0.4),
            ),
            customdata=np.stack([core.get("event_id", pd.Series([""] * len(core))), core["area_m2"]], axis=-1),
            hovertemplate="event=%{customdata[0]}<br>area=%{customdata[1]:.0f} m2<extra></extra>",
            name="landslides",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if not lai_event.empty:
        fig.add_trace(
            go.Histogram(
                x=lai_event["pre_lai_mean"],
                nbinsx=30,
                marker=dict(color="#b2182b", opacity=0.72, line=dict(color="white", width=0.4)),
                showlegend=False,
                hovertemplate="pre-event LAI=%{x:.3f}<br>count=%{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    if not slope_event.empty:
        fig.add_trace(
            go.Histogram(
                x=slope_event["slope_mean"],
                nbinsx=30,
                marker=dict(color="#4d4d4d", opacity=0.72, line=dict(color="white", width=0.4)),
                showlegend=False,
                hovertemplate="slope=%{x:.2f}<br>count=%{y}<extra></extra>",
            ),
            row=1,
            col=3,
        )
    else:
        fig.add_annotation(
            text="slope feature table not found",
            x=0.5,
            y=0.5,
            xref="x3 domain",
            yref="y3 domain",
            showarrow=False,
            font=dict(size=13, color="#555555"),
        )

    labels = [
        "(a) Map of landslides",
        "(b) Pre-event LAI distribution",
        "(c) Slope distribution",
    ]
    for idx, text in enumerate(labels, start=1):
        fig.add_annotation(
            text=text,
            x=0.02,
            y=0.96,
            xref=f"x{idx} domain" if idx > 1 else "x domain",
            yref=f"y{idx} domain" if idx > 1 else "y domain",
            showarrow=False,
            align="left",
            font=dict(size=13, color="#111"),
        )
    fig.update_layout(
        title_text="Fig. 4 Landslide distribution patterns",
        template="plotly_white",
        width=1280,
        height=520,
        margin=dict(l=70, r=110, t=85, b=115),
        font=dict(family="Arial", size=12, color="#111"),
    )
    fig.update_xaxes(showline=True, mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, mirror=True, ticks="outside")
    fig.update_xaxes(title_text="Longitude / X", row=1, col=1)
    fig.update_yaxes(title_text="Latitude / Y", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="Pre-event LAI", row=1, col=2)
    fig.update_xaxes(title_text="Slope", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    _add_caption(
        fig,
        "Fig. 4. Landslide distribution patterns in Jiangwan. The histograms use one mean value per landslide rather than pixel-level samples. Pre-event LAI is measured at t-1 in the core zone; slope is static and read from the recovery-rate feature table when available.",
    )
    path = figures_dir / "Dashboard_04_Landslide_Distribution_Patterns.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return [path]


def plot_drop_class_distribution(summary: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """把 0-20% 到 80-100% 五档灾时下降分级画成堆叠占比图。"""
    figures_dir = ensure_dir(figures_dir)
    if summary.empty:
        return []
    df = summary.copy()
    required = {"outcome", "zone_name", "event_year", "drop_class", "n_landslides", "total_area_m2"}
    if not required.issubset(df.columns):
        return []
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
    df["n_landslides"] = pd.to_numeric(df["n_landslides"], errors="coerce").fillna(0.0)
    df["total_area_m2"] = pd.to_numeric(df["total_area_m2"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["event_year"]).copy()
    if df.empty:
        return []
    df["event_year"] = df["event_year"].astype(int)
    df["drop_class"] = pd.Categorical(df["drop_class"], categories=DROP_CLASS_ORDER, ordered=True)

    outcomes = _ordered_outcomes(df["outcome"].dropna().unique().tolist())
    zones = _ordered_zones(df["zone_name"].dropna().unique().tolist())
    years = sorted(df["event_year"].dropna().unique().tolist())
    if not outcomes or not zones or not years:
        return []

    # 补齐空等级，确保每个子图都是同一套 5 档颜色。
    full_index = pd.MultiIndex.from_product(
        [outcomes, zones, years, DROP_CLASS_ORDER],
        names=["outcome", "zone_name", "event_year", "drop_class"],
    )
    df = (
        df.set_index(["outcome", "zone_name", "event_year", "drop_class"])
        .reindex(full_index)
        .reset_index()
    )
    for col in ["n_landslides", "total_area_m2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    totals = (
        df.groupby(["outcome", "zone_name", "event_year"], as_index=False)["n_landslides"]
        .sum()
        .rename(columns={"n_landslides": "year_total"})
    )
    df = df.merge(totals, on=["outcome", "zone_name", "event_year"], how="left")
    df["share_percent"] = np.where(df["year_total"] > 0, df["n_landslides"] / df["year_total"] * 100.0, 0.0)

    fig = make_subplots(rows=len(outcomes), cols=len(zones), horizontal_spacing=0.055, vertical_spacing=0.10)
    for r, outcome in enumerate(outcomes, start=1):
        for c, zone in enumerate(zones, start=1):
            for drop_class in DROP_CLASS_ORDER:
                sdf = df[
                    (df["outcome"] == outcome)
                    & (df["zone_name"] == zone)
                    & (df["drop_class"].astype(str) == drop_class)
                ].sort_values("event_year")
                if sdf.empty:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=sdf["event_year"],
                        y=sdf["share_percent"],
                        marker=dict(color=DROP_CLASS_COLORS[drop_class], line=dict(color="rgba(0,0,0,0.35)", width=0.3)),
                        name=drop_class,
                        legendgroup=drop_class,
                        showlegend=(r == 1 and c == len(zones)),
                        customdata=np.stack(
                            [
                                sdf["n_landslides"],
                                sdf["year_total"],
                                sdf["total_area_m2"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "Year=%{x}<br>"
                            "Drop class=" + drop_class + "<br>"
                            "Share=%{y:.1f}%<br>"
                            "n=%{customdata[0]:.0f}/%{customdata[1]:.0f}<br>"
                            "area=%{customdata[2]:.0f} m2<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
            fig.add_annotation(
                text=f"({chr(96 + (r - 1) * len(zones) + c)}) {outcome} | {_zone_label(zone)}",
                x=0.03,
                y=0.95,
                xref=f"x{(r - 1) * len(zones) + c} domain" if (r - 1) * len(zones) + c > 1 else "x domain",
                yref=f"y{(r - 1) * len(zones) + c} domain" if (r - 1) * len(zones) + c > 1 else "y domain",
                showarrow=False,
                align="left",
                font=dict(size=10, color="#111"),
            )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        height=max(760, 280 * len(outcomes)),
        width=max(1180, 300 * len(zones)),
        margin=dict(l=70, r=170, t=78, b=130),
        title=dict(
            text="Event-year drop class distribution",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial, Microsoft YaHei, sans-serif", color="#111"),
        ),
        legend=dict(title_text="Drop class", x=1.01, y=0.5, xanchor="left", yanchor="middle"),
    )
    fig.update_xaxes(showline=True, mirror=True, ticks="outside", dtick=1)
    fig.update_yaxes(showline=True, mirror=True, ticks="outside", range=[0, 100])
    fig.update_xaxes(title_text="Event year", row=len(outcomes), col=max(1, len(zones) // 2))
    fig.update_yaxes(title_text="Landslide share (%)", row=max(1, len(outcomes) // 2), col=1)
    _add_caption(
        fig,
        "Observed event-year drop class distribution in town Jiangwan. Each subplot stacks the five t-1 to t0 drop classes by event year; this is an individual-landslide diagnostic, not a DID ATT estimate.",
    )
    path = figures_dir / "Dashboard_04_Drop_Class_Distribution.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return [path]


def plot_drop_class_histograms(metrics: pd.DataFrame, figures_dir: str | Path) -> list[Path]:
    """绘制灾时下降百分比分布曲线，同时给出 Count 和 Share 两种纵轴口径。"""
    figures_dir = ensure_dir(figures_dir)
    if metrics.empty:
        return []
    required = {"outcome", "zone_name", "drop_percent", "event_id", "area_m2"}
    if not required.issubset(metrics.columns):
        return []

    df = metrics.copy()
    df["drop_percent"] = pd.to_numeric(df["drop_percent"], errors="coerce")
    df["area_m2"] = pd.to_numeric(df["area_m2"], errors="coerce")
    df = df.dropna(subset=["drop_percent"])
    if df.empty:
        return []
    df["drop_percent"] = df["drop_percent"].clip(lower=0, upper=100)
    # 图上采用“下降为负”的变化口径，便于和 Event Study 的 ATT% 方向保持一致。
    df["change_percent"] = -df["drop_percent"]

    outcomes = _ordered_outcomes(df["outcome"].dropna().unique().tolist())
    if not outcomes:
        return []

    x_min, x_max = -100.0, 0.0
    # 5% 分箱比 2% 分箱更稳，不会把曲线画成很多细碎毛刺。
    bins = np.arange(x_min, x_max + 5.0, 5.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    zone_styles = {
        "no_inner_core": {"color": "#4d4d4d", "dash": "solid", "width": 2.7, "label": "no-inner core"},
        "core_inner": {"color": "#111111", "dash": "solid", "width": 3.2, "label": "inner core"},
        "ring_0_10m": {"color": "#2b8cbe", "dash": "dash", "width": 2.7, "label": "0-10 m"},
        "ring_10_20m": {"color": "#f03b20", "dash": "dot", "width": 2.7, "label": "10-20 m"},
    }
    zone_order = ["no_inner_core", "core_inner", "ring_0_10m", "ring_10_20m"]
    fig = make_subplots(
        rows=len(outcomes),
        cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.10,
        vertical_spacing=0.11,
        column_widths=[0.50, 0.50],
    )
    curve_rows: list[dict[str, object]] = []
    letters = list("abcdefghijklmnopqrstuvwxyz")

    def event_changes_for_zone(outcome_df: pd.DataFrame, zone: str, all_events: pd.Index) -> pd.DataFrame:
        """返回某个 zone 的事件级灾时变化；缺失 zone 不再补 0，避免 0% 人工尖峰。"""
        if zone == "no_inner_core":
            inner_events = pd.Index(
                outcome_df.loc[
                    outcome_df["zone_name"].astype(str) == "core_inner",
                    "event_id",
                ].dropna().astype(str).unique()
            )
            target_events = all_events.difference(inner_events)
            zdf = outcome_df[
                (outcome_df["zone_name"].astype(str) == "core")
                & (outcome_df["event_id"].astype(str).isin(target_events))
            ].copy()
        else:
            zdf = outcome_df[outcome_df["zone_name"].astype(str) == zone].copy()
        if zdf.empty:
            return pd.DataFrame(columns=["event_id", "change_percent"])
        return (
            zdf.dropna(subset=["event_id"])
            .assign(event_id=lambda x: x["event_id"].astype(str))
            .groupby("event_id", as_index=False)["change_percent"]
            .mean()
        )

    for r, outcome in enumerate(outcomes, start=1):
        y_max_count = 0.0
        y_max_share = 0.0
        outcome_df = df[df["outcome"] == outcome].copy()
        all_events = pd.Index(outcome_df["event_id"].dropna().astype(str).unique())
        denominator = max(float(len(all_events)), 1.0)
        for zone in zone_order:
            event_changes = event_changes_for_zone(outcome_df, zone, all_events)
            range_df = event_changes[(event_changes["change_percent"] >= x_min) & (event_changes["change_percent"] <= x_max)].copy()
            if range_df.empty:
                continue
            counts, _ = np.histogram(range_df["change_percent"], bins=bins)
            counts = counts.astype(float)
            shares = counts / denominator * 100.0
            y_max_count = max(y_max_count, float(np.nanmax(counts)) if len(counts) else 0.0)
            y_max_share = max(y_max_share, float(np.nanmax(shares)) if len(shares) else 0.0)
            style = zone_styles[zone]
            zone_label = str(style["label"])
            for i, center in enumerate(bin_centers):
                curve_rows.append(
                    {
                        "outcome": outcome,
                        "zone_key": zone,
                        "zone_label": zone_label,
                        "bin_left": float(bins[i]),
                        "bin_right": float(bins[i + 1]),
                        "bin_center": float(center),
                        "count": float(counts[i]),
                        "share_percent": float(shares[i]),
                        "denominator_all_events": float(denominator),
                        "observed_zone_events": int(len(event_changes)),
                    }
                )

            for col, y_values, y_name, show_legend in [
                (1, counts, "Count", r == 1),
                (2, shares, "Share", False),
            ]:
                smooth_x, smooth_y = _shape_preserving_curve(
                    bin_centers.tolist(),
                    y_values.tolist(),
                    points_per_segment=14,
                )
                fig.add_trace(
                    go.Scatter(
                        x=smooth_x,
                        y=smooth_y,
                        mode="lines",
                        line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                        fill="tozeroy",
                        fillcolor=_hex_to_rgba(str(style["color"]), 0.08),
                        name=zone_label,
                        legendgroup=zone,
                        showlegend=show_legend,
                        hoverinfo="skip",
                    ),
                    row=r,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=y_values,
                        mode="markers",
                        marker=dict(size=7.5, color=style["color"], line=dict(color="white", width=0.65)),
                        name=zone_label,
                        legendgroup=zone,
                        showlegend=False,
                        customdata=np.stack(
                            [bins[:-1], bins[1:], counts, shares, np.repeat(denominator, len(bin_centers))],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "Zone=%{fullData.name}<br>"
                            "Change bin=%{customdata[0]:.0f} to %{customdata[1]:.0f}%<br>"
                            f"{y_name}=%{{y:.2f}}<br>"
                            "Count=%{customdata[2]:.0f}<br>"
                            "Share=%{customdata[3]:.2f}%<br>"
                            "all events=%{customdata[4]:.0f}<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=col,
                )
        fig.add_annotation(
            text=f"<b>({letters[(r - 1) * 2]})</b> {outcome} | Count",
            x=0.02,
            y=0.94,
            xref=f"x{(r - 1) * 2 + 1} domain" if r > 1 else "x domain",
            yref=f"y{(r - 1) * 2 + 1} domain" if r > 1 else "y domain",
            showarrow=False,
            align="left",
            font=dict(size=13, color="#111"),
        )
        fig.add_annotation(
            text=f"<b>({letters[(r - 1) * 2 + 1]})</b> {outcome} | Share",
            x=0.02,
            y=0.94,
            xref=f"x{(r - 1) * 2 + 2} domain",
            yref=f"y{(r - 1) * 2 + 2} domain",
            showarrow=False,
            align="left",
            font=dict(size=13, color="#111"),
        )
        fig.update_yaxes(range=[0, max(y_max_count * 1.18, 5.0)], row=r, col=1)
        fig.update_yaxes(range=[0, max(y_max_share * 1.18, 2.0)], row=r, col=2)

    source_dir = ensure_dir(Path(figures_dir).parent / "06_figure_source_data" / "Dashboard_05_Drop_Class_Distribution_Curves")
    safe_to_csv(
        pd.DataFrame(curve_rows),
        source_dir / "drop_class_distribution_curves_count_share.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig.update_layout(
        template="plotly_white",
        height=max(760, 300 * len(outcomes)),
        width=1420,
        margin=dict(l=78, r=165, t=78, b=135),
        title=dict(
            text="Event-year drop percentage distributions: Count and Share",
            x=0.0,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=20, family="Arial, Microsoft YaHei, sans-serif", color="#111"),
        ),
        legend=dict(title_text="Zone", x=1.01, y=0.5, xanchor="left", yanchor="middle"),
    )
    fig.update_xaxes(
        showline=True,
        mirror=True,
        ticks="outside",
        range=[x_min, x_max],
        tickmode="array",
        tickvals=[-100, -80, -60, -40, -20, 0],
    )
    fig.update_yaxes(showline=True, mirror=True, ticks="outside", rangemode="tozero")
    fig.update_xaxes(title_text="Change from t-1 to t0 (%)", row=len(outcomes), col=1)
    fig.update_xaxes(title_text="Change from t-1 to t0 (%)", row=len(outcomes), col=2)
    fig.update_yaxes(title_text="Count", row=max(1, len(outcomes) // 2), col=1)
    fig.update_yaxes(title_text="Share of all landslides (%)", row=max(1, len(outcomes) // 2), col=2)
    _add_caption(
        fig,
        "Observed event-year change percentage distributions in town Jiangwan. The x axis is the t-1 to t0 percentage change, so ecological decline is negative. Left panels show raw counts; right panels show the share of all landslide events for each outcome. Missing zones are not forced to 0% change. Curves are shape-preserving interpolations through 5% histogram-bin sample points.",
    )
    path = figures_dir / "Dashboard_05_Drop_Class_Distribution_Curves.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return [path]


def run_individual_drop_diagnostics(
    panel: pd.DataFrame,
    zones_path: str | Path,
    output_dir: str | Path,
    figures_dir: str | Path,
    value_column: str = "mean",
    template_rasters_by_year: dict[int, str | Path] | None = None,
    study_area_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """输出单滑坡突降分级 CSV、GPKG 和 HTML 诊断图。"""
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(figures_dir)
    metrics, summary, value_summary = build_individual_drop_metrics(panel=panel, value_column=value_column)

    safe_to_csv(metrics, output_dir / "individual_zone_drop_metrics.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(summary, output_dir / "individual_zone_drop_class_summary.csv", index=False, encoding="utf-8-sig")
    safe_to_csv(value_summary, output_dir / "individual_zone_value_distribution_summary.csv", index=False, encoding="utf-8-sig")

    paths: list[Path] = []
    paths.extend(
        plot_landslide_distribution_patterns(
            panel=panel,
            zones_path=zones_path,
            output_dir=output_dir,
            figures_dir=figures_dir,
            value_column=value_column,
            study_area_path=study_area_path,
        )
    )

    if logger is not None:
        logger.info(
            "Wrote compact individual diagnostics: metrics=%s rows, summary=%s rows, figures=%s",
            len(metrics),
            len(summary),
            len(paths),
        )
    return {
        "metrics": metrics,
        "summary": summary,
        "value_summary": value_summary,
        "raster_manifest": pd.DataFrame(),
        "polygons": pd.DataFrame(),
    }

