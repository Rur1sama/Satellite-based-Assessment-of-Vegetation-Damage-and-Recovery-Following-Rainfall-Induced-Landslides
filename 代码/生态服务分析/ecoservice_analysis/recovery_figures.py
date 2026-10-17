from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .area_response import (
    OUTCOME_COLORS,
    _add_caption,
    _axis_domain_ref,
    _fixed_area_bins_labels,
    _ordered_outcomes,
    _series_label,
    _smooth_xy,
    _write_html_with_fallback,
)
from .io_utils import ensure_dir


RECOVERY_LEVEL_PERCENT = 90
RECOVERY_ATT_THRESHOLD_PERCENT = -(100 - RECOVERY_LEVEL_PERCENT)
RELATIVE_RECOVERY_YEARS = list(range(0, 6))
# 自然断点结果按老师建议规整为便于解释的 10 度坡度等级。
# 最后一档使用 90 度作为封闭上界，图表标签仍显示为 >30°。
SLOPE_CLASS_BREAKS = [0.0, 10.0, 20.0, 30.0, 90.0]
# 使用 en dash 表示数值范围，避免 Excel 将例如“10-20”自动识别为日期。
SLOPE_CLASS_LABELS = ["0–10", "10–20", "20–30", ">30"]


def _outcome_label(outcome: str) -> str:
    """Return display label for an outcome."""
    return {"LAI_SR2": "LAI", "FVC": "FVC", "GPP": "GPP"}.get(str(outcome), str(outcome))


def _extract_event_slope_table(
    events_path: str | Path,
    slope_raster_path: str | Path,
    output_path: str | Path,
    all_touched: bool = False,
) -> pd.DataFrame:
    '''
    根据当前滑坡事件几何，从坡度栅格重新提取每个事件的平均坡度。
    不复用旧制图表中的 event_id，避免滑坡重新矢量化后编号与几何错配。

    :param events_path: 当前流程生成的滑坡事件矢量路径。
    :param slope_raster_path: 坡度栅格路径，像元值单位为度。
    :param output_path: 事件级坡度结果表输出路径。
    :param all_touched: 是否统计所有被滑坡边界触碰到的坡度像元。
    :return: 包含 event_id、event_year、slope_mean 和 slope_n_pixels 的数据表。
    '''
    import geopandas as gpd
    import rasterio

    from .panel import _zonal_stats_for_geometry

    events_path = Path(events_path)
    slope_raster_path = Path(slope_raster_path)
    output_path = Path(output_path)
    if not events_path.exists():
        raise FileNotFoundError(f"Missing current landslide events: {events_path}")
    if not slope_raster_path.exists():
        raise FileNotFoundError(f"Missing slope raster: {slope_raster_path}")

    events = gpd.read_file(events_path)
    required = {"event_id", "geometry"}
    missing_columns = required.difference(events.columns)
    if missing_columns:
        raise ValueError(f"Landslide events lack required columns: {sorted(missing_columns)}")
    if events.empty:
        raise ValueError(f"No landslide events found in {events_path}")
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate event_id values found in {events_path}")

    with rasterio.open(slope_raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Slope raster has no CRS: {slope_raster_path}")
        if events.crs is None:
            raise ValueError(f"Landslide events have no CRS: {events_path}")
        slope_events = events.to_crs(src.crs) if events.crs != src.crs else events.copy()
        rows: list[dict[str, object]] = []
        for row in slope_events.itertuples(index=False):
            stats = _zonal_stats_for_geometry(
                src,
                row.geometry,
                band=1,
                all_touched=all_touched,
            )
            rows.append(
                {
                    "event_id": str(row.event_id),
                    "event_year": getattr(row, "event_year", np.nan),
                    "slope_mean": stats["mean"],
                    "slope_n_pixels": stats["n_pixels"],
                }
            )

    result = pd.DataFrame(rows)
    result["slope_mean"] = pd.to_numeric(result["slope_mean"], errors="coerce")
    result["slope_n_pixels"] = (
        pd.to_numeric(result["slope_n_pixels"], errors="coerce").fillna(0).astype(int)
    )
    missing_slope = result["slope_mean"].isna() | (result["slope_n_pixels"] <= 0)
    if missing_slope.any():
        examples = result.loc[missing_slope, "event_id"].head(10).tolist()
        raise ValueError(
            f"Slope extraction failed for {int(missing_slope.sum())} current events; examples: {examples}"
        )

    ensure_dir(output_path.parent)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def _load_event_slope_table(
    results_dir: Path,
    events_path: str | Path | None = None,
    slope_raster_path: str | Path | None = None,
    all_touched: bool = False,
) -> pd.DataFrame:
    '''
    为恢复分析构建与当前滑坡事件严格一致的坡度表。

    :param results_dir: DID 结果目录。
    :param events_path: 当前滑坡事件矢量路径；为空时使用流程默认路径。
    :param slope_raster_path: 坡度栅格路径；为空时使用 Output/Slope.tif。
    :param all_touched: 是否统计所有被滑坡边界触碰到的坡度像元。
    :return: 当前全部滑坡事件的平均坡度表。
    '''
    current_events_path = (
        Path(events_path)
        if events_path is not None
        else results_dir.parent / "01_landslide_events.gpkg"
    )
    current_slope_path = (
        Path(slope_raster_path)
        if slope_raster_path is not None
        else results_dir.parent.parent / "Slope.tif"
    )
    return _extract_event_slope_table(
        events_path=current_events_path,
        slope_raster_path=current_slope_path,
        output_path=results_dir / "event_slope_table.csv",
        all_touched=all_touched,
    )


def _attach_area_and_slope_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Attach area and slope bins to an event-level table."""
    result = df.copy()
    area_bins, area_labels = _fixed_area_bins_labels()
    result["area_bin"] = pd.cut(
        result["landslide_core_area_m2"],
        bins=area_bins,
        labels=area_labels,
        include_lowest=True,
        right=True,
    ).astype(str)
    result["area_bin_order"] = result["area_bin"].map({label: i for i, label in enumerate(area_labels)})

    slope = pd.to_numeric(result.get("slope_mean"), errors="coerce")
    finite = slope[np.isfinite(slope)]
    if finite.empty:
        result["slope_bin"] = "slope missing"
        result["slope_bin_order"] = 999
        return result

    result["slope_bin"] = pd.cut(
        slope,
        bins=SLOPE_CLASS_BREAKS,
        labels=SLOPE_CLASS_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    result["slope_bin_order"] = result["slope_bin"].map(
        {label: i for i, label in enumerate(SLOPE_CLASS_LABELS)}
    )
    result["slope_bin_left_deg"] = result["slope_bin_order"].map(
        {i: SLOPE_CLASS_BREAKS[i] for i in range(len(SLOPE_CLASS_LABELS))}
    )
    result["slope_bin_right_deg"] = result["slope_bin_order"].map(
        {i: SLOPE_CLASS_BREAKS[i + 1] for i in range(len(SLOPE_CLASS_LABELS))}
    )
    result.loc[slope.isna(), "slope_bin"] = "slope missing"
    result.loc[slope.isna(), "slope_bin_order"] = 999
    result.loc[slope.isna(), ["slope_bin_left_deg", "slope_bin_right_deg"]] = np.nan
    return result


def build_recovery_event_table(
    individual_att: pd.DataFrame,
    slope_table: pd.DataFrame | None = None,
    recovery_threshold_percent: float = RECOVERY_ATT_THRESHOLD_PERCENT,
    relative_years: list[int] | None = None,
) -> pd.DataFrame:
    """
    鏋勫缓鍗曟粦鍧℃仮澶嶈揪鏍囪〃銆?
    褰撳墠瀹氫箟涓衡€滄仮澶嶅埌鏃犳粦鍧″弽浜嬪疄姘村钩鐨?RECOVERY_LEVEL_PERCENT%鈥濓細
    ATT% >= RECOVERY_ATT_THRESHOLD_PERCENT銆?    鏈揪鍒伴槇鍊肩殑婊戝潯淇濈暀鍦ㄨ〃涓紝骞舵爣璁颁负 censored锛岄伩鍏嶇绾垮浘鎶婂畠浠潤榛樺垹闄ゃ€?    """
    if individual_att.empty:
        return pd.DataFrame()

    years = sorted(int(v) for v in (relative_years or RELATIVE_RECOVERY_YEARS) if int(v) >= 0)
    df = individual_att.copy()
    df = df[df["outcome"].astype(str).isin(_ordered_outcomes(df["outcome"].dropna().unique().tolist()))].copy()
    for col in ["relative_year", "individual_att_percent", "landslide_core_area_m2", "event_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_id"] = df["event_id"].astype(str)
    if "size_class" not in df.columns:
        df["size_class"] = np.where(df["landslide_core_area_m2"] < 1000.0, "small", "large")
    df["size_class"] = df["size_class"].astype(str)
    df = df.dropna(subset=["outcome", "event_id", "relative_year", "individual_att_percent", "landslide_core_area_m2"])
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (outcome, event_id), curve in df.groupby(["outcome", "event_id"], dropna=True):
        curve = curve.sort_values("relative_year").copy()
        event_rows = curve[curve["relative_year"] == 0]
        if event_rows.empty:
            continue
        event_row = event_rows.iloc[0]
        event_att = float(event_row["individual_att_percent"])
        if not np.isfinite(event_att) or event_att > float(recovery_threshold_percent):
            continue
        post = curve[curve["relative_year"].isin(years)].copy()
        first_hit = post[post["individual_att_percent"] >= float(recovery_threshold_percent)]
        first_year = float(first_hit.iloc[0]["relative_year"]) if not first_hit.empty else np.nan
        latest_observed = float(post["relative_year"].max()) if not post.empty else np.nan
        base = {
            "outcome": str(outcome),
            "event_id": str(event_id),
            "event_year": int(event_row["event_year"]) if pd.notna(event_row["event_year"]) else np.nan,
            "size_class": str(event_row["size_class"]),
            "landslide_core_area_m2": float(event_row["landslide_core_area_m2"]),
            "event_att_percent": event_att,
            "recovery_level_percent": int(RECOVERY_LEVEL_PERCENT),
            "recovery_att_threshold_percent": float(recovery_threshold_percent),
            "recovery_threshold_mode": "counterfactual_level",
            "first_recovered_relative_year": first_year,
            "is_recovered_by_t5": bool(pd.notna(first_year) and first_year <= max(years)),
            "is_censored_unrecovered": bool(pd.isna(first_year)),
            "latest_observed_relative_year": latest_observed,
        }
        for year in years:
            base[f"recovered_by_t{year}"] = bool(pd.notna(first_year) and first_year <= year)
        rows.append(base)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if slope_table is not None and not slope_table.empty:
        slope = slope_table[["event_id", "slope_mean"]].drop_duplicates("event_id").copy()
        slope["event_id"] = slope["event_id"].astype(str)
        result = result.merge(slope, on="event_id", how="left")
    else:
        result["slope_mean"] = np.nan
    return _attach_area_and_slope_bins(result).sort_values(["outcome", "event_year", "event_id"])


def _recovery_time_summary(event_table: pd.DataFrame, bin_col: str, order_col: str) -> pd.DataFrame:
    """Summarize recovery time by area or slope bins."""
    if event_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (outcome, bin_label), chunk in event_table.groupby(["outcome", bin_col], dropna=False):
        recovered = pd.to_numeric(chunk["first_recovered_relative_year"], errors="coerce").dropna()
        total = int(len(chunk))
        rows.append(
            {
                "outcome": outcome,
                "bin_type": bin_col.replace("_bin", ""),
                "bin_label": str(bin_label),
                "bin_order": int(pd.to_numeric(chunk[order_col], errors="coerce").dropna().min())
                if pd.to_numeric(chunk[order_col], errors="coerce").notna().any()
                else 999,
                "n_total_landslides": total,
                "n_recovered": int(len(recovered)),
                "n_unrecovered_censored": int(total - len(recovered)),
                "recovered_share_percent": float(len(recovered) / total * 100.0) if total else np.nan,
                "censored_share_percent": float((total - len(recovered)) / total * 100.0) if total else np.nan,
                "recovery_time_mean_year": float(recovered.mean()) if len(recovered) else np.nan,
                "recovery_time_q05_year": float(recovered.quantile(0.05)) if len(recovered) else np.nan,
                "recovery_time_q25_year": float(recovered.quantile(0.25)) if len(recovered) else np.nan,
                "recovery_time_median_year": float(recovered.median()) if len(recovered) else np.nan,
                "recovery_time_q75_year": float(recovered.quantile(0.75)) if len(recovered) else np.nan,
                "recovery_time_q95_year": float(recovered.quantile(0.95)) if len(recovered) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["outcome", "bin_order", "bin_label"])


def _cumulative_recovery_summary(event_table: pd.DataFrame, group_cols: list[str], order_cols: list[str]) -> pd.DataFrame:
    """Summarize cumulative recovery share by group and relative year."""
    if event_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, chunk in event_table.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for order_col in order_cols:
            if order_col in chunk.columns:
                vals = pd.to_numeric(chunk[order_col], errors="coerce").dropna()
                base[order_col] = int(vals.min()) if not vals.empty else 999
        total = int(len(chunk))
        first_year = pd.to_numeric(chunk["first_recovered_relative_year"], errors="coerce")
        for year in RELATIVE_RECOVERY_YEARS:
            n = int((first_year <= year).sum())
            rows.append(
                {
                    **base,
                    "relative_year": int(year),
                    "n_total_landslides": total,
                    "n_recovered_cumulative": n,
                    "cumulative_recovered_share_percent": float(n / total * 100.0) if total else np.nan,
                }
            )
    sort_cols = [col for col in ["outcome", *order_cols, *group_cols, "relative_year"] if col in rows[0]] if rows else []
    return pd.DataFrame(rows).sort_values(sort_cols)


def _iqr_filtered_mean(values: pd.Series, multiplier: float = 3.0) -> dict[str, float]:
    """Return the IQR-filtered mean and diagnostics."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    raw_n = int(len(vals))
    if raw_n == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr_low": np.nan,
            "iqr_high": np.nan,
            "std": np.nan,
            "se": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_raw": 0,
            "n_used": 0,
            "n_removed": 0,
        }
    q25 = float(vals.quantile(0.25))
    q75 = float(vals.quantile(0.75))
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 0:
        low, high = q25, q75
        kept = vals.copy()
    else:
        low = q25 - float(multiplier) * iqr
        high = q75 + float(multiplier) * iqr
        kept = vals[(vals >= low) & (vals <= high)]
    if kept.empty:
        kept = vals.copy()
    kept_n = int(len(kept))
    mean = float(kept.mean())
    std = float(kept.std(ddof=1)) if kept_n > 1 else 0.0
    se = std / float(np.sqrt(kept_n)) if kept_n > 0 else np.nan
    ci_half = 1.96 * se if np.isfinite(se) else np.nan
    return {
        "mean": mean,
        "median": float(kept.median()),
        "q25": q25,
        "q75": q75,
        "iqr_low": float(low),
        "iqr_high": float(high),
        "std": std,
        "se": se,
        "ci95_low": mean - ci_half if np.isfinite(ci_half) else np.nan,
        "ci95_high": mean + ci_half if np.isfinite(ci_half) else np.nan,
        "n_raw": raw_n,
        "n_used": kept_n,
        "n_removed": int(raw_n - kept_n),
    }


def _prepare_individual_att_with_bins(individual_att: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """Join individual ATT trajectories with event-level bins."""
    if individual_att.empty or event_table.empty:
        return pd.DataFrame()
    df = individual_att.copy()
    for col in ["relative_year", "individual_att_percent", "event_year", "landslide_core_area_m2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_id"] = df["event_id"].astype(str)
    df["outcome"] = df["outcome"].astype(str)
    df = df[df["relative_year"].isin(RELATIVE_RECOVERY_YEARS)].copy()
    meta_cols = [
        "outcome",
        "event_id",
        "size_class",
        "landslide_core_area_m2",
        "slope_mean",
        "area_bin",
        "area_bin_order",
        "slope_bin",
        "slope_bin_order",
    ]
    meta = event_table[[col for col in meta_cols if col in event_table.columns]].drop_duplicates(["outcome", "event_id"])
    meta["event_id"] = meta["event_id"].astype(str)
    meta["outcome"] = meta["outcome"].astype(str)
    df = df.drop(columns=[col for col in meta.columns if col not in {"outcome", "event_id"} and col in df.columns])
    return df.merge(meta, on=["outcome", "event_id"], how="inner")


def _build_first_year_recovery_metrics(individual_att: pd.DataFrame, slope_table: pd.DataFrame) -> pd.DataFrame:
    """Build per-landslide first-year recovery metrics from cached individual ATT%."""
    if individual_att.empty:
        return pd.DataFrame()
    needed = {"outcome", "event_id", "relative_year", "individual_att_percent"}
    if not needed.issubset(individual_att.columns):
        return pd.DataFrame()
    df = individual_att.copy()
    df["event_id"] = df["event_id"].astype(str)
    df["outcome"] = df["outcome"].astype(str)
    for col in ["relative_year", "individual_att_percent", "landslide_core_area_m2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pivot = (
        df[df["relative_year"].isin([0, 1])]
        .pivot_table(
            index=["outcome", "event_id"],
            columns="relative_year",
            values="individual_att_percent",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={0: "event_att_percent", 1: "t1_att_percent"})
    )
    meta_cols = ["outcome", "event_id", "landslide_core_area_m2", "event_year"]
    meta = df[[col for col in meta_cols if col in df.columns]].drop_duplicates(["outcome", "event_id"])
    metrics = pivot.merge(meta, on=["outcome", "event_id"], how="left")
    if "landslide_core_area_m2" in metrics.columns:
        metrics["landslide_core_area_m2"] = pd.to_numeric(metrics["landslide_core_area_m2"], errors="coerce")
    else:
        metrics["landslide_core_area_m2"] = np.nan

    if slope_table is not None and not slope_table.empty:
        slope = slope_table[["event_id", "slope_mean"]].drop_duplicates("event_id").copy()
        slope["event_id"] = slope["event_id"].astype(str)
        metrics = metrics.merge(slope, on="event_id", how="left")
    else:
        metrics["slope_mean"] = np.nan
    metrics = _attach_area_and_slope_bins(metrics)
    denom = 0.0 - pd.to_numeric(metrics["event_att_percent"], errors="coerce")
    numer = pd.to_numeric(metrics["t1_att_percent"], errors="coerce") - pd.to_numeric(
        metrics["event_att_percent"], errors="coerce"
    )
    metrics["first_year_recovery_percent_raw"] = np.where(
        denom > 0,
        numer / denom * 100.0,
        np.nan,
    )
    metrics["valid_first_year_recovery"] = (
        (pd.to_numeric(metrics["event_att_percent"], errors="coerce") <= RECOVERY_ATT_THRESHOLD_PERCENT)
        & pd.to_numeric(metrics["t1_att_percent"], errors="coerce").notna()
        & pd.to_numeric(metrics["first_year_recovery_percent_raw"], errors="coerce").notna()
    )
    metrics["first_year_recovery_percent_capped"] = pd.to_numeric(
        metrics["first_year_recovery_percent_raw"], errors="coerce"
    ).clip(lower=0.0, upper=100.0)
    metrics["over_recovery"] = pd.to_numeric(metrics["first_year_recovery_percent_raw"], errors="coerce") > 100.0
    metrics["no_recovery"] = pd.to_numeric(metrics["first_year_recovery_percent_raw"], errors="coerce") <= 0.0
    return metrics.sort_values(["outcome", "event_id"])


def _first_year_recovery_bin_summary(metrics: pd.DataFrame, bin_col: str, order_col: str) -> pd.DataFrame:
    '''
    按面积或坡度分组汇总首年恢复率。

    :param metrics: 逐滑坡首年恢复率数据。
    :param bin_col: 分组标签列名。
    :param order_col: 分组排序列名。
    :return: 每个生态指标、每个分组对应的一行统计结果。
    '''
    if metrics.empty or bin_col not in metrics.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    source = metrics[metrics["valid_first_year_recovery"]].copy()
    for (outcome, bin_label), chunk in source.groupby(["outcome", bin_col], dropna=True):
        y = pd.to_numeric(chunk["first_year_recovery_percent_capped"], errors="coerce").dropna()
        raw = pd.to_numeric(chunk["first_year_recovery_percent_raw"], errors="coerce").dropna()
        if y.empty:
            continue
        order_vals = pd.to_numeric(chunk.get(order_col, pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "outcome": outcome,
                "bin_type": bin_col,
                "bin_label": bin_label,
                "bin_order": int(order_vals.min()) if not order_vals.empty else 999,
                "n_landslides": int(len(y)),
                "mean_first_year_recovery_percent": float(y.mean()),
                "median_first_year_recovery_percent": float(y.median()),
                "p05_first_year_recovery_percent": float(y.quantile(0.05)),
                "p25_first_year_recovery_percent": float(y.quantile(0.25)),
                "p75_first_year_recovery_percent": float(y.quantile(0.75)),
                "p95_first_year_recovery_percent": float(y.quantile(0.95)),
                "over_recovery_share_percent": float((raw > 100.0).mean() * 100.0) if not raw.empty else np.nan,
                "no_recovery_share_percent": float((raw <= 0.0).mean() * 100.0) if not raw.empty else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["outcome", "bin_order", "bin_label"])


def _robust_mean_trajectory(
    source: pd.DataFrame,
    group_cols: list[str],
    order_cols: list[str],
) -> pd.DataFrame:
    """Build IQR-filtered mean ATT trajectories by relative year."""
    if source.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, chunk in source.groupby(["outcome", *group_cols, "relative_year"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_names = ["outcome", *group_cols, "relative_year"]
        base = dict(zip(key_names, keys))
        for order_col in order_cols:
            if order_col in chunk.columns:
                vals = pd.to_numeric(chunk[order_col], errors="coerce").dropna()
                base[order_col] = int(vals.min()) if not vals.empty else 999
        stats = _iqr_filtered_mean(chunk["individual_att_percent"], multiplier=3.0)
        rows.append(
            {
                **base,
                "robust_mean_att_percent": stats["mean"],
                "median_att_percent": stats["median"],
                "q25_att_percent": stats["q25"],
                "q75_att_percent": stats["q75"],
                "iqr_filter_low_percent": stats["iqr_low"],
                "iqr_filter_high_percent": stats["iqr_high"],
                "std_att_percent": stats["std"],
                "se_att_percent": stats["se"],
                "ci95_low_att_percent": stats["ci95_low"],
                "ci95_high_att_percent": stats["ci95_high"],
                "n_raw_landslides": stats["n_raw"],
                "n_used_landslides": stats["n_used"],
                "n_removed_by_3iqr": stats["n_removed"],
            }
        )
    sort_cols = [col for col in ["outcome", *order_cols, *group_cols, "relative_year"] if col in rows[0]] if rows else []
    return pd.DataFrame(rows).sort_values(sort_cols)


def _log_recovery_fit(
    curve: pd.DataFrame,
    y_col: str,
    threshold: float = RECOVERY_ATT_THRESHOLD_PERCENT,
    min_year: float = 0.0,
    max_year: float = 5.0,
    grid_points: int = 181,
) -> dict[str, object]:
    '''
    使用对数函数拟合恢复轨迹，并计算阈值交叉年份与拟合误差。

    :param curve: 包含相对年份及待拟合指标的恢复轨迹表。
    :param y_col: 作为因变量的列名。
    :param threshold: 恢复判定所用的 ATT 百分比阈值。
    :param min_year: 参与拟合的最小相对年份。
    :param max_year: 参与拟合的最大相对年份。
    :param grid_points: 拟合曲线输出网格的节点数。
    :return: 拟合曲线、参数、R²、RMSE、阈值交叉年份及状态。
    '''
    empty = {
        "x": np.array([], dtype=float),
        "y": np.array([], dtype=float),
        "intercept": np.nan,
        "slope": np.nan,
        "r2": np.nan,
        "rmse": np.nan,
        "crossing_year_raw": np.nan,
        "crossing_year_in_window": np.nan,
        "status": "fit_failed",
    }
    if y_col not in curve.columns or "relative_year" not in curve.columns:
        return empty
    data = (
        curve[["relative_year", y_col]]
        .rename(columns={y_col: "y"})
        .assign(relative_year=lambda d: pd.to_numeric(d["relative_year"], errors="coerce"), y=lambda d: pd.to_numeric(d["y"], errors="coerce"))
        .dropna()
        .groupby("relative_year", as_index=False)["y"]
        .mean()
        .sort_values("relative_year")
    )
    data = data[(data["relative_year"] >= min_year) & (data["relative_year"] <= max_year)]
    if len(data) < 2:
        return empty
    x = data["relative_year"].astype(float).to_numpy()
    y = data["y"].astype(float).to_numpy()
    z = np.log1p(x)
    z_mean = float(np.mean(z))
    y_mean = float(np.mean(y))
    denom = float(np.sum((z - z_mean) ** 2))
    if not np.isfinite(denom) or denom <= 0:
        return empty
    slope = float(np.sum((z - z_mean) * (y - y_mean)) / denom)
    intercept = float(y_mean - slope * z_mean)
    x_grid = np.linspace(float(min_year), float(max_year), int(max(grid_points, 2)))
    y_grid = intercept + slope * np.log1p(x_grid)
    y_hat = intercept + slope * z
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(y))) if len(y) > 0 else np.nan
    y_min = float(intercept + slope * np.log1p(min_year))
    y_max = float(intercept + slope * np.log1p(max_year))
    crossing_raw = np.nan
    crossing_window = np.nan
    status = "not_recovered_by_t5"
    if np.isfinite(slope) and abs(float(slope)) > 1e-12:
        crossing_raw = float(np.exp((threshold - intercept) / slope) - 1.0)
        if y_min >= threshold:
            crossing_window = float(min_year)
            status = "recovered_at_t0"
        elif y_max >= threshold and min_year <= crossing_raw <= max_year:
            crossing_window = crossing_raw
            status = "recovered_by_log_fit"
        elif np.isfinite(crossing_raw) and crossing_raw > max_year:
            status = "not_recovered_by_t5"
        else:
            status = "no_valid_crossing"
    return {
        "x": x_grid,
        "y": y_grid,
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
        "crossing_year_raw": crossing_raw,
        "crossing_year_in_window": crossing_window,
        "status": status,
    }


def _finite_float(value: object) -> float:
    """Return a finite float or nan."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _format_crossing_label(mean_fit: dict[str, object], lower_fit: dict[str, object], upper_fit: dict[str, object]) -> str:
    """Compact visible label for the fitted recovery threshold crossing."""
    mean_year = _finite_float(mean_fit.get("crossing_year_in_window"))
    lower_year = _finite_float(lower_fit.get("crossing_year_in_window"))
    upper_year = _finite_float(upper_fit.get("crossing_year_in_window"))
    if not np.isfinite(mean_year):
        return ">5 yr"
    ci_vals = [v for v in [lower_year, upper_year] if np.isfinite(v)]
    if len(ci_vals) == 2:
        return f"{mean_year:.1f} yr [{min(ci_vals):.1f}, {max(ci_vals):.1f}]"
    return f"{mean_year:.1f} yr"


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba()."""
    text = str(color).lstrip("#")
    if len(text) != 6:
        return f"rgba(90,90,90,{alpha})"
    try:
        r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return f"rgba(90,90,90,{alpha})"
    return f"rgba({r},{g},{b},{alpha})"


def _mean_fit_recovery_summary(
    trajectory: pd.DataFrame,
    group_cols: list[str],
    order_cols: list[str],
    threshold: float = RECOVERY_ATT_THRESHOLD_PERCENT,
) -> pd.DataFrame:
    """Fit grouped mean trajectories and summarize recovery-year crossings."""
    if trajectory.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, chunk in trajectory.groupby(["outcome", *group_cols], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(["outcome", *group_cols], keys))
        for order_col in order_cols:
            if order_col in chunk.columns:
                vals = pd.to_numeric(chunk[order_col], errors="coerce").dropna()
                base[order_col] = int(vals.min()) if not vals.empty else 999
        curve = chunk.dropna(subset=["relative_year", "robust_mean_att_percent"]).sort_values("relative_year")
        mean_fit = _log_recovery_fit(curve, "robust_mean_att_percent", threshold=threshold)
        lower_fit = _log_recovery_fit(curve, "ci95_low_att_percent", threshold=threshold)
        upper_fit = _log_recovery_fit(curve, "ci95_high_att_percent", threshold=threshold)
        full_fit = _log_recovery_fit(curve, "robust_mean_att_percent", threshold=0.0)
        full_lower_fit = _log_recovery_fit(curve, "ci95_low_att_percent", threshold=0.0)
        full_upper_fit = _log_recovery_fit(curve, "ci95_high_att_percent", threshold=0.0)
        ci_values = [
            float(v)
            for v in [
                lower_fit["crossing_year_in_window"],
                upper_fit["crossing_year_in_window"],
            ]
            if np.isfinite(v)
        ]
        rows.append(
            {
                **base,
                "recovery_level_percent": RECOVERY_LEVEL_PERCENT,
                "recovery_att_threshold_percent": threshold,
                "recovery_threshold_mode": "counterfactual_level",
                "mean_fit_recovered_relative_year": mean_fit["crossing_year_in_window"],
                "mean_fit_status": mean_fit["status"],
                "log_fit_intercept": mean_fit["intercept"],
                "log_fit_slope": mean_fit["slope"],
                "log_fit_r2": mean_fit["r2"],
                "log_fit_rmse": mean_fit["rmse"],
                "log_fit_recovered_relative_year": mean_fit["crossing_year_in_window"],
                "log_fit_recovered_relative_year_raw": mean_fit["crossing_year_raw"],
                "log_fit_recovery_year_ci_low": min(ci_values) if len(ci_values) == 2 else np.nan,
                "log_fit_recovery_year_ci_high": max(ci_values) if len(ci_values) == 2 else np.nan,
                "log_fit_lower_curve_recovered_relative_year": lower_fit["crossing_year_in_window"],
                "log_fit_upper_curve_recovered_relative_year": upper_fit["crossing_year_in_window"],
                "log_fit_lower_curve_status": lower_fit["status"],
                "log_fit_upper_curve_status": upper_fit["status"],
                "log_fit_full_recovery_threshold_percent": 0.0,
                "log_fit_full_recovery_relative_year_raw": full_fit["crossing_year_raw"],
                "log_fit_full_recovery_status": full_fit["status"],
                "log_fit_full_recovery_ci_low_raw": np.nanmin(
                    [
                        _finite_float(full_lower_fit["crossing_year_raw"]),
                        _finite_float(full_upper_fit["crossing_year_raw"]),
                    ]
                )
                if any(
                    np.isfinite(v)
                    for v in [
                        _finite_float(full_lower_fit["crossing_year_raw"]),
                        _finite_float(full_upper_fit["crossing_year_raw"]),
                    ]
                )
                else np.nan,
                "log_fit_full_recovery_ci_high_raw": np.nanmax(
                    [
                        _finite_float(full_lower_fit["crossing_year_raw"]),
                        _finite_float(full_upper_fit["crossing_year_raw"]),
                    ]
                )
                if any(
                    np.isfinite(v)
                    for v in [
                        _finite_float(full_lower_fit["crossing_year_raw"]),
                        _finite_float(full_upper_fit["crossing_year_raw"]),
                    ]
                )
                else np.nan,
                "n_time_nodes": int(curve["relative_year"].nunique()) if not curve.empty else 0,
                "mean_t0_att_percent": float(curve.loc[curve["relative_year"] == 0, "robust_mean_att_percent"].iloc[0])
                if (curve["relative_year"] == 0).any()
                else np.nan,
                "mean_t5_att_percent": float(curve.loc[curve["relative_year"] == 5, "robust_mean_att_percent"].iloc[0])
                if (curve["relative_year"] == 5).any()
                else np.nan,
            }
        )
    sort_cols = [col for col in ["outcome", *order_cols, *group_cols] if col in rows[0]] if rows else []
    return pd.DataFrame(rows).sort_values(sort_cols)


def build_recovery_summary_tables(
    results_dir: str | Path,
    events_path: str | Path | None = None,
    slope_raster_path: str | Path | None = None,
    slope_all_touched: bool = False,
) -> dict[str, pd.DataFrame]:
    '''
    构建 Fig.6 和 Fig.7 的恢复分析标准化数据表。

    :param results_dir: DID 结果目录。
    :param events_path: 当前滑坡事件矢量路径。
    :param slope_raster_path: 坡度栅格路径。
    :param slope_all_touched: 是否统计所有被滑坡边界触碰到的坡度像元。
    :return: 各类恢复分析数据表组成的字典。
    '''
    results_dir = Path(results_dir)
    individual_path = results_dir / "area_individual_att_percent.csv"
    if not individual_path.exists():
        raise FileNotFoundError(f"Missing individual ATT table: {individual_path}")
    individual_att = pd.read_csv(individual_path)
    slope_table = _load_event_slope_table(
        results_dir,
        events_path=events_path,
        slope_raster_path=slope_raster_path,
        all_touched=slope_all_touched,
    )
    event_table = build_recovery_event_table(individual_att, slope_table=slope_table)
    individual_with_bins = _prepare_individual_att_with_bins(individual_att, event_table)
    size_mean = _robust_mean_trajectory(individual_with_bins, ["size_class"], [])
    size_mean_fit = _mean_fit_recovery_summary(size_mean, ["size_class"], [])
    first_year_metrics = _build_first_year_recovery_metrics(individual_att, slope_table)
    first_year_area = _first_year_recovery_bin_summary(first_year_metrics, "area_bin", "area_bin_order")
    first_year_slope = _first_year_recovery_bin_summary(first_year_metrics, "slope_bin", "slope_bin_order")
    return {
        "event_slope_table": slope_table,
        "fig07_mean_size_trajectory": size_mean,
        "fig07_mean_fit_size_summary": size_mean_fit,
        "fig08_first_year_recovery_metrics": first_year_metrics,
        "fig08_first_year_recovery_area_bin_summary": first_year_area,
        "fig08_first_year_recovery_slope_bin_summary": first_year_slope,
    }


def _write_figure_source_table(table: pd.DataFrame, output_path: str | Path) -> Path:
    '''
    将单张图所需的完整长表写入独立的制图数据目录。

    :param table: 已整理的制图长表。
    :param output_path: CSV 输出路径。
    :return: 实际写入的文件路径。
    '''
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    try:
        table.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_updated{output_path.suffix}")
        table.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def plot_fig06_recovery_time_boxplots(
    event_table: pd.DataFrame,
    area_summary: pd.DataFrame,
    slope_summary: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig.6A recovery-time boxplots."""
    figures_dir = ensure_dir(figures_dir)
    if event_table.empty:
        return []
    outcomes = _ordered_outcomes(event_table["outcome"].dropna().unique().tolist())
    if not outcomes:
        return []
    fig = make_subplots(
        rows=len(outcomes),
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.13,
        column_titles=["By area class", "By slope class"],
    )
    letters = list("abcdefghijklmnopqrstuvwxyz")
    for row, outcome in enumerate(outcomes, start=1):
        base = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        odf = event_table[event_table["outcome"] == outcome].copy()
        for col, (bin_col, order_col, summary) in enumerate(
            [
                ("area_bin", "area_bin_order", area_summary),
                ("slope_bin", "slope_bin_order", slope_summary),
            ],
            start=1,
        ):
            ss = summary[(summary["outcome"] == outcome)].copy()
            if not ss.empty:
                ss = ss.sort_values("bin_order")
                visible = ss[ss["n_recovered"].fillna(0).astype(float) >= 3].copy()
                for _, stat in visible.iterrows():
                    bin_label = str(stat["bin_label"])
                    fig.add_trace(
                        go.Box(
                            x=[bin_label],
                            q1=[float(stat["recovery_time_q25_year"])],
                            median=[float(stat["recovery_time_median_year"])],
                            q3=[float(stat["recovery_time_q75_year"])],
                            lowerfence=[float(stat["recovery_time_q05_year"])],
                            upperfence=[float(stat["recovery_time_q95_year"])],
                            name=_outcome_label(outcome),
                            legendgroup=outcome,
                            showlegend=False,
                            boxpoints=False,
                            marker=dict(color=base),
                            line=dict(color=base, width=1.5),
                            fillcolor=_hex_to_rgba(base, 0.16),
                            width=0.48,
                            hovertemplate=(
                                f"bin={bin_label}<br>median=%{{median:.1f}} yr<br>"
                                "q25=%{q1:.1f} yr<br>q75=%{q3:.1f} yr<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=visible["bin_label"].astype(str),
                        y=visible["recovery_time_median_year"],
                        mode="lines+markers",
                        name="median recovery time",
                        legendgroup=outcome,
                        showlegend=False,
                        line=dict(color=base, width=2.0),
                        marker=dict(color=base, size=5),
                        hovertemplate="bin=%{x}<br>median recovery time=%{y:.1f} yr<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            if not ss.empty:
                fig.add_trace(
                    go.Scatter(
                        x=ss["bin_label"],
                        y=np.repeat(5.35, len(ss)),
                        mode="markers",
                        marker=dict(
                            color="rgba(70,70,70,0.72)",
                            size=np.clip(ss["censored_share_percent"].fillna(0) / 4.0 + 4.0, 4.0, 16.0),
                            symbol="x",
                        ),
                        name="unrecovered/censored",
                        legendgroup="censored",
                        showlegend=(row == 1 and col == 2),
                        customdata=np.stack(
                            [ss["n_total_landslides"], ss["n_unrecovered_censored"], ss["censored_share_percent"]],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "bin=%{x}<br>total=%{customdata[0]:.0f}<br>"
                            "unrecovered=%{customdata[1]:.0f}<br>censored=%{customdata[2]:.1f}%<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            panel_idx = (row - 1) * 2 + col
            fig.add_annotation(
                text=f"<b>({letters[panel_idx - 1]})</b> {_outcome_label(outcome)}",
                x=0.015,
                y=0.95,
                xref=_axis_domain_ref("x", panel_idx),
                yref=_axis_domain_ref("y", panel_idx),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
    fig.update_yaxes(range=[-0.2, 5.6], tickmode="array", tickvals=RELATIVE_RECOVERY_YEARS, showline=True, mirror=True, ticks="outside")
    fig.update_xaxes(showline=True, mirror=True, ticks="outside", tickangle=35)
    fig.update_yaxes(
        title_text=f"Years to {RECOVERY_LEVEL_PERCENT}% recovery",
        row=max(1, len(outcomes) // 2),
        col=1,
    )
    fig.update_layout(
        title_text=f"Fig. 6A Recovery time to {RECOVERY_LEVEL_PERCENT}% counterfactual level",
        template="plotly_white",
        width=1280,
        height=max(760, len(outcomes) * 280 + 160),
        margin=dict(l=86, r=185, t=88, b=150),
        legend=dict(x=1.01, y=0.5, xanchor="left", yanchor="middle"),
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=12, color="#111111"),
    )
    _add_caption(
        fig,
        f"Fig. 6A. Recovery time in Jiangwan. A landslide is recovered when ATT% reaches at least "
        f"{RECOVERY_ATT_THRESHOLD_PERCENT:.0f}%, equivalent to {RECOVERY_LEVEL_PERCENT}% of its no-landslide "
        "counterfactual level. Boxplots use recovered landslides only; "
        "x markers summarize unrecovered/censored samples within each bin.",
        y=-0.14,
    )
    return [_write_html_with_fallback(fig, Path(figures_dir) / "Dashboard_06A_Recovery_Time_Boxplots.html")]


def plot_fig08_first_year_recovery_boxplots(
    metrics: pd.DataFrame,
    area_summary: pd.DataFrame,
    slope_summary: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig.6C: first-year recovery boxplots split by area and slope bins."""
    figures_dir = ensure_dir(figures_dir)
    if metrics.empty:
        return []
    source = metrics[metrics["valid_first_year_recovery"]].copy()
    if source.empty:
        return []
    outcomes = _ordered_outcomes(source["outcome"].dropna().unique().tolist())
    if not outcomes:
        return []
    fig = make_subplots(
        rows=len(outcomes),
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.13,
        column_titles=["By area class", "By slope class"],
    )
    letters = list("abcdefghijklmnopqrstuvwxyz")
    plot_specs = [
        ("area_bin", "area_bin_order", area_summary, "Core area class (m2)"),
        ("slope_bin", "slope_bin_order", slope_summary, "Slope class (degree)"),
    ]
    legend_added = False
    for row, outcome in enumerate(outcomes, start=1):
        color = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        for col_idx, (bin_col, order_col, summary, x_title) in enumerate(plot_specs, start=1):
            subset = source[(source["outcome"] == outcome) & source[bin_col].notna()].copy()
            if subset.empty:
                continue
            summary_subset = summary[summary["outcome"] == outcome].copy() if not summary.empty else pd.DataFrame()
            if summary_subset.empty:
                continue
            summary_subset = summary_subset.sort_values(["bin_order", "bin_label"])
            box_labels: list[str] = []
            for _, stat in summary_subset.iterrows():
                if int(stat.get("n_landslides", 0)) < 5:
                    continue
                bin_label = str(stat["bin_label"])
                box_labels.append(bin_label)
                fig.add_trace(
                    go.Box(
                        x=[bin_label],
                        q1=[float(stat["p25_first_year_recovery_percent"])],
                        median=[float(stat["median_first_year_recovery_percent"])],
                        q3=[float(stat["p75_first_year_recovery_percent"])],
                        lowerfence=[float(stat["p05_first_year_recovery_percent"])],
                        upperfence=[float(stat["p95_first_year_recovery_percent"])],
                        name="5-95% recovery box",
                        boxpoints=False,
                        line=dict(color=color, width=1.3),
                        marker=dict(color=color),
                        fillcolor=_hex_to_rgba(color, 0.18),
                        width=0.48,
                        showlegend=(row == 1 and col_idx == 1 and len(box_labels) == 1),
                        legendgroup="recovery_box",
                        hovertemplate=f"class={bin_label}<br>median=%{{median:.1f}}%<extra></extra>",
                    ),
                    row=row,
                    col=col_idx,
                )
            if box_labels:
                summary_subset = summary_subset[summary_subset["bin_label"].astype(str).isin(box_labels)].copy()
            if not summary_subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=summary_subset["bin_label"].astype(str),
                        y=summary_subset["median_first_year_recovery_percent"],
                        mode="lines+markers",
                        name="median first-year recovery",
                        line=dict(color=color, width=2.4),
                        marker=dict(color=color, size=5.5, line=dict(color="white", width=0.6)),
                        showlegend=(row == 1 and col_idx == 2),
                        legendgroup="median_first_year_recovery",
                        hovertemplate="class=%{x}<br>median recovery=%{y:.1f}%<extra></extra>",
                    ),
                    row=row,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=summary_subset["bin_label"].astype(str),
                        y=summary_subset["over_recovery_share_percent"],
                        mode="lines+markers",
                        name="over-recovery share",
                        line=dict(color="#111111", width=2.2),
                        marker=dict(color="#111111", size=5),
                        showlegend=not legend_added,
                    ),
                    row=row,
                    col=col_idx,
                )
                legend_added = True
            panel_idx = (row - 1) * 2 + col_idx
            fig.add_annotation(
                text=f"<b>({letters[panel_idx - 1]})</b> {_outcome_label(outcome)}",
                x=0.015,
                y=0.95,
                xref=_axis_domain_ref("x", panel_idx),
                yref=_axis_domain_ref("y", panel_idx),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
            fig.update_xaxes(title_text=x_title if row == len(outcomes) else None, row=row, col=col_idx)
    fig.update_yaxes(range=[0, 100], showline=True, mirror=True, ticks="outside")
    fig.update_xaxes(showline=True, mirror=True, ticks="outside", tickangle=35)
    fig.update_yaxes(title_text="First-year recovery / share (%)", row=max(1, len(outcomes) // 2), col=1)
    fig.update_layout(
        title_text="Fig. 6C First-year recovery by area and slope classes",
        template="plotly_white",
        width=1320,
        height=max(820, len(outcomes) * 300 + 170),
        margin=dict(l=92, r=190, t=90, b=155),
        legend=dict(x=1.01, y=0.52, xanchor="left", yanchor="middle"),
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=12, color="#111111"),
    )
    _add_caption(
        fig,
        "Fig. 6C. First-year recovery in Jiangwan. First-year recovery is "
        "(ATT(t+1) - ATT(t0)) / (0 - ATT(t0)) * 100 for landslides with event-year ATT below the recovery threshold; "
        "boxplots show capped 0-100% recovery values, and the black line shows the share of raw values above 100%.",
        y=-0.14,
    )
    return [_write_html_with_fallback(fig, Path(figures_dir) / "Dashboard_08_FirstYear_Recovery_Boxplots.html")]


def plot_fig06_cumulative_curves(
    area_cum: pd.DataFrame,
    slope_cum: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig.6B log-fitted recovery curves by area and slope bins."""
    figures_dir = ensure_dir(figures_dir)
    if area_cum.empty and slope_cum.empty:
        return []
    outcomes = _ordered_outcomes(
        list(area_cum.get("outcome", pd.Series(dtype=str)).dropna().unique())
        + list(slope_cum.get("outcome", pd.Series(dtype=str)).dropna().unique())
    )
    fig = make_subplots(
        rows=len(outcomes),
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.13,
        column_titles=["By area class", "By slope class"],
    )
    letters = list("abcdefghijklmnopqrstuvwxyz")
    for row, outcome in enumerate(outcomes, start=1):
        base = OUTCOME_COLORS.get(outcome, "#4d4d4d")
        for col, (df, bin_col, order_col) in enumerate(
            [(area_cum, "area_bin", "area_bin_order"), (slope_cum, "slope_bin", "slope_bin_order")],
            start=1,
        ):
            sub = df[df["outcome"] == outcome].copy()
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[min(RELATIVE_RECOVERY_YEARS), max(RELATIVE_RECOVERY_YEARS)],
                    y=[RECOVERY_ATT_THRESHOLD_PERCENT, RECOVERY_ATT_THRESHOLD_PERCENT],
                    mode="lines",
                    line=dict(color="rgba(80,80,80,0.52)", width=1.1, dash="dash"),
                    name=f"{RECOVERY_LEVEL_PERCENT}% threshold",
                    legendgroup="recovery_threshold",
                    showlegend=(row == 1 and col == 2),
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            ordered_bins = (
                sub[[bin_col, order_col]]
                .drop_duplicates()
                .sort_values([order_col, bin_col])[bin_col]
                .astype(str)
                .tolist()
            )
            for idx, bin_label in enumerate(ordered_bins):
                sdf = sub[sub[bin_col].astype(str) == str(bin_label)].sort_values("relative_year")
                if sdf.empty:
                    continue
                shade = 0.15 + 0.68 * (idx / max(len(ordered_bins) - 1, 1))
                color = _blend_hex(base, "#f7f7f7", shade)
                fit = _log_recovery_fit(sdf, "robust_mean_att_percent")
                lower_fit = _log_recovery_fit(sdf, "ci95_low_att_percent")
                upper_fit = _log_recovery_fit(sdf, "ci95_high_att_percent")
                if len(lower_fit["x"]) and len(upper_fit["x"]):
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([upper_fit["x"], lower_fit["x"][::-1]]),
                            y=np.concatenate([upper_fit["y"], lower_fit["y"][::-1]]),
                            mode="lines",
                            fill="toself",
                            fillcolor=_hex_to_rgba(color, 0.10),
                            line=dict(color="rgba(255,255,255,0)", width=0),
                            legendgroup=f"{bin_col}_{bin_label}",
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )
                fig.add_trace(
                    go.Scatter(
                        x=fit["x"],
                        y=fit["y"],
                        mode="lines",
                        line=dict(color=color, width=2.2),
                        name=str(bin_label),
                        legendgroup=f"{bin_col}_{bin_label}",
                        showlegend=(row == 1),
                        hovertemplate=f"{_outcome_label(outcome)} {bin_label}<br>relative year=%{{x:.2f}}<br>log-fitted ATT=%{{y:.1f}}%<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                crossing_year = _finite_float(fit["crossing_year_in_window"])
                if np.isfinite(crossing_year):
                    fig.add_trace(
                        go.Scatter(
                            x=[crossing_year],
                            y=[RECOVERY_ATT_THRESHOLD_PERCENT],
                            mode="markers",
                            marker=dict(symbol="diamond", color=color, size=6.8, line=dict(color="white", width=0.7)),
                            legendgroup=f"{bin_col}_{bin_label}",
                            showlegend=False,
                            hovertemplate=(
                                f"{_outcome_label(outcome)} {bin_label}<br>"
                                f"{RECOVERY_LEVEL_PERCENT}% recovery crossing="
                                f"{_format_crossing_label(fit, lower_fit, upper_fit)}<br>"
                                f"log R2={_finite_float(fit['r2']):.2f}<br>"
                                f"RMSE={_finite_float(fit['rmse']):.2f} percentage points<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=col,
                    )
                custom = np.stack(
                    [
                        sdf["n_raw_landslides"],
                        sdf["n_used_landslides"],
                        sdf["n_removed_by_3iqr"],
                        sdf["ci95_low_att_percent"],
                        sdf["ci95_high_att_percent"],
                    ],
                    axis=-1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=sdf["relative_year"],
                        y=sdf["robust_mean_att_percent"],
                        mode="markers",
                        marker=dict(color=color, size=4.8),
                        showlegend=False,
                        customdata=custom,
                        hovertemplate=(
                            f"{_outcome_label(outcome)} {bin_label}<br>"
                            "relative year=%{x}<br>robust mean ATT=%{y:.1f}%<br>"
                            "95% CI=[%{customdata[3]:.1f}, %{customdata[4]:.1f}]<br>"
                            "raw n=%{customdata[0]:.0f}<br>used n=%{customdata[1]:.0f}<br>"
                            "removed by 3IQR=%{customdata[2]:.0f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            panel_idx = (row - 1) * 2 + col
            fig.add_annotation(
                text=f"<b>({letters[panel_idx - 1]})</b> {_outcome_label(outcome)}",
                x=0.015,
                y=0.95,
                xref=_axis_domain_ref("x", panel_idx),
                yref=_axis_domain_ref("y", panel_idx),
                showarrow=False,
                font=dict(size=13, color="black"),
            )
    fig.update_xaxes(tickmode="array", tickvals=RELATIVE_RECOVERY_YEARS, showline=True, mirror=True, ticks="outside")
    fig.update_yaxes(range=[-100, 20], showline=True, mirror=True, ticks="outside", zeroline=True, zerolinecolor="#333333")
    fig.update_yaxes(title_text="Robust mean DID ATT (%)", row=max(1, len(outcomes) // 2), col=1)
    fig.update_layout(
        title_text="Fig. 6B Log-fitted recovery year by area and slope bins",
        template="plotly_white",
        width=1280,
        height=max(760, len(outcomes) * 280 + 160),
        margin=dict(l=86, r=220, t=88, b=150),
        legend=dict(x=1.01, y=0.5, xanchor="left", yanchor="middle", bgcolor="rgba(255,255,255,0.86)"),
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=12, color="#111111"),
    )
    _add_caption(
        fig,
        "Fig. 6B. Log-fitted recovery by area and slope bins. Points are group mean DID ATT% values after removing "
        "values outside 3 x IQR. Lines fit ATT% = a + b ln(1 + t); shaded bands fit 95% CI bounds, and diamonds "
        f"mark intersections with the {RECOVERY_LEVEL_PERCENT}% counterfactual-level recovery threshold.",
        y=-0.14,
    )
    return [_write_html_with_fallback(fig, Path(figures_dir) / "Dashboard_06B_Cumulative_Recovery_Curves.html")]


def plot_fig07_cumulative_size_bars(
    size_cum: pd.DataFrame,
    figures_dir: str | Path,
) -> list[Path]:
    """Fig.7 log-fitted recovery curves by landslide size."""
    figures_dir = ensure_dir(figures_dir)
    if size_cum.empty:
        return []
    df = size_cum.copy()
    outcomes = _ordered_outcomes(df["outcome"].dropna().unique().tolist())
    projection_years: list[float] = []
    for _, chunk in df.groupby(["outcome", "size_class"], dropna=False):
        full_fit = _log_recovery_fit(chunk.sort_values("relative_year"), "robust_mean_att_percent", threshold=0.0)
        year = _finite_float(full_fit["crossing_year_raw"])
        if np.isfinite(year) and year >= 0:
            projection_years.append(year)
    max_projection_year = min(max([5.0, *projection_years]) + 0.5, 12.0)
    fig = make_subplots(rows=1, cols=len(outcomes), horizontal_spacing=0.08)
    colors = {"small": "#2b8cbe", "large": "#d95f0e"}
    dashes = {"small": "dash", "large": "solid"}
    letters = list("abcdefghijklmnopqrstuvwxyz")
    for col, outcome in enumerate(outcomes, start=1):
        odf = df[df["outcome"] == outcome].copy()
        fig.add_trace(
            go.Scatter(
                x=[min(RELATIVE_RECOVERY_YEARS), max_projection_year],
                y=[RECOVERY_ATT_THRESHOLD_PERCENT, RECOVERY_ATT_THRESHOLD_PERCENT],
                mode="lines",
                line=dict(color="rgba(80,80,80,0.52)", width=1.1, dash="dash"),
                name=f"{RECOVERY_LEVEL_PERCENT}% threshold",
                legendgroup="recovery_threshold",
                showlegend=(col == len(outcomes)),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[min(RELATIVE_RECOVERY_YEARS), max_projection_year],
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(30,30,30,0.55)", width=1.15, dash="dot"),
                name="counterfactual level",
                legendgroup="counterfactual_level",
                showlegend=(col == len(outcomes)),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[5, 5],
                y=[-100, 20],
                mode="lines",
                line=dict(color="rgba(90,90,90,0.35)", width=1.0, dash="dash"),
                name="observed t5",
                legendgroup="observed_t5",
                showlegend=(col == len(outcomes)),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )
        for size_class in ["small", "large"]:
            sdf = odf[odf["size_class"].astype(str) == size_class].sort_values("relative_year")
            if sdf.empty:
                continue
            fit = _log_recovery_fit(sdf, "robust_mean_att_percent", max_year=max_projection_year)
            lower_fit = _log_recovery_fit(sdf, "ci95_low_att_percent", max_year=max_projection_year)
            upper_fit = _log_recovery_fit(sdf, "ci95_high_att_percent", max_year=max_projection_year)
            full_fit = _log_recovery_fit(sdf, "robust_mean_att_percent", threshold=0.0, max_year=max_projection_year)
            full_lower_fit = _log_recovery_fit(sdf, "ci95_low_att_percent", threshold=0.0, max_year=max_projection_year)
            full_upper_fit = _log_recovery_fit(sdf, "ci95_high_att_percent", threshold=0.0, max_year=max_projection_year)
            color = colors.get(size_class, "#bdbdbd")
            if len(lower_fit["x"]) and len(upper_fit["x"]):
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([upper_fit["x"], lower_fit["x"][::-1]]),
                        y=np.concatenate([upper_fit["y"], lower_fit["y"][::-1]]),
                        mode="lines",
                        fill="toself",
                        fillcolor=_hex_to_rgba(color, 0.13),
                        line=dict(color="rgba(255,255,255,0)", width=0),
                        name=f"{_series_label(size_class)} 95% CI",
                        legendgroup=size_class,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
            full_year = _finite_float(full_fit["crossing_year_raw"])
            if np.isfinite(full_year) and 0 <= full_year <= max_projection_year:
                ci_vals = [
                    _finite_float(full_lower_fit["crossing_year_raw"]),
                    _finite_float(full_upper_fit["crossing_year_raw"]),
                ]
                ci_vals = [v for v in ci_vals if np.isfinite(v)]
                full_label = (
                    f"{full_year:.1f} yr [{min(ci_vals):.1f}, {max(ci_vals):.1f}]"
                    if len(ci_vals) == 2
                    else f"{full_year:.1f} yr"
                )
                fig.add_trace(
                    go.Scatter(
                        x=[full_year],
                        y=[0],
                        mode="markers+text",
                        marker=dict(symbol="star", color=color, size=10.5, line=dict(color="white", width=0.8)),
                        text=[full_label],
                        textposition="top center" if size_class == "large" else "bottom center",
                        textfont=dict(size=10, color=color),
                        name=f"{_series_label(size_class)} projected full recovery",
                        legendgroup=size_class,
                        showlegend=False,
                        hovertemplate=(
                            f"{_outcome_label(outcome)} {_series_label(size_class)}<br>"
                            f"projected full recovery={full_label}<br>"
                            "threshold ATT=0%<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
            fig.add_trace(
                go.Scatter(
                    x=fit["x"],
                    y=fit["y"],
                    mode="lines",
                    line=dict(color=color, width=2.6, dash=dashes.get(size_class, "solid")),
                    name=_series_label(size_class),
                    legendgroup=size_class,
                    showlegend=(col == len(outcomes)),
                    hovertemplate=(
                        f"{_outcome_label(outcome)} {_series_label(size_class)}<br>"
                        "relative year=%{x:.2f}<br>log-fitted ATT=%{y:.1f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            crossing_year = _finite_float(fit["crossing_year_in_window"])
            if np.isfinite(crossing_year):
                label = _format_crossing_label(fit, lower_fit, upper_fit)
                fig.add_trace(
                    go.Scatter(
                        x=[crossing_year],
                        y=[RECOVERY_ATT_THRESHOLD_PERCENT],
                        mode="markers+text",
                        marker=dict(symbol="diamond", color=color, size=8.5, line=dict(color="white", width=0.8)),
                        text=[label],
                        textposition="top center" if size_class == "large" else "bottom center",
                        textfont=dict(size=10, color=color),
                        name=f"{_series_label(size_class)} recovery year",
                        legendgroup=size_class,
                        showlegend=False,
                        hovertemplate=(
                            f"{_outcome_label(outcome)} {_series_label(size_class)}<br>"
                            f"{RECOVERY_LEVEL_PERCENT}% recovery crossing={label}<br>"
                            f"log R2={_finite_float(fit['r2']):.2f}<br>"
                            f"RMSE={_finite_float(fit['rmse']):.2f} percentage points<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
            custom = np.stack(
                [
                    sdf["n_raw_landslides"],
                    sdf["n_used_landslides"],
                    sdf["n_removed_by_3iqr"],
                    sdf["ci95_low_att_percent"],
                    sdf["ci95_high_att_percent"],
                ],
                axis=-1,
            )
            fig.add_trace(
                go.Scatter(
                    x=sdf["relative_year"],
                    y=sdf["robust_mean_att_percent"],
                    mode="markers",
                    marker=dict(color=color, size=6.2, line=dict(color="white", width=0.6)),
                    legendgroup=size_class,
                    showlegend=False,
                    customdata=custom,
                    hovertemplate=(
                        f"{_outcome_label(outcome)} {_series_label(size_class)}<br>"
                        "relative year=%{x}<br>robust mean ATT=%{y:.1f}%<br>"
                        "95% CI=[%{customdata[3]:.1f}, %{customdata[4]:.1f}]<br>"
                        "raw n=%{customdata[0]:.0f}<br>used n=%{customdata[1]:.0f}<br>"
                        "removed by 3IQR=%{customdata[2]:.0f}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.add_annotation(
            text=f"<b>({letters[col - 1]})</b> {_outcome_label(outcome)}",
            x=0.02,
            y=0.96,
            xref=_axis_domain_ref("x", col),
            yref=_axis_domain_ref("y", col),
            showarrow=False,
            font=dict(size=13, color="black"),
        )
    fig.update_xaxes(
        range=[0, max_projection_year],
        tickmode="array",
        tickvals=list(range(0, int(np.ceil(max_projection_year)) + 1)),
        title_text="",
        showline=True,
        mirror=True,
        ticks="outside",
    )
    if outcomes:
        fig.update_xaxes(title_text="Relative year", row=1, col=max(1, int(np.ceil(len(outcomes) / 2))))
    fig.update_yaxes(range=[-100, 20], showline=True, mirror=True, ticks="outside", zeroline=True, zerolinecolor="#333333")
    fig.update_yaxes(title_text="Robust mean DID ATT (%)", row=1, col=1)
    fig.update_layout(
        title=dict(text="Fig. 7 Log-fitted recovery projection by landslide size", x=0.02, xanchor="left"),
        template="plotly_white",
        width=max(980, 360 * len(outcomes) + 220),
        height=560,
        margin=dict(l=82, r=205, t=92, b=175),
        legend=dict(x=1.01, y=0.5, xanchor="left", yanchor="middle"),
        font=dict(family="Arial, Microsoft YaHei, sans-serif", size=12, color="#111111"),
    )
    _add_caption(
        fig,
        f"Fig. 7. Log-fitted recovery projection by landslide size. Points are 3 x IQR filtered mean DID ATT% values "
        f"at t0-t5; lines fit ATT% = a + b ln(1 + t) and are extrapolated beyond t5 when needed. Diamonds show "
        f"the fitted intersection with the {RECOVERY_LEVEL_PERCENT}% counterfactual-level recovery threshold; stars show the projected "
        "future year when the trajectory reaches the no-landslide counterfactual level (ATT = 0%).",
        y=-0.25,
    )
    return [_write_html_with_fallback(fig, Path(figures_dir) / "Dashboard_07_Cumulative_Recovery_Rate.html")]


def _blend_hex(color: str, other: str, ratio: float) -> str:
    """Return a gradient color scale for an outcome."""
    def parse(value: str) -> tuple[int, int, int]:
        text = str(value).lstrip("#")
        if len(text) != 6:
            return (80, 80, 80)
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))

    r1, g1, b1 = parse(color)
    r2, g2, b2 = parse(other)
    ratio = min(max(float(ratio), 0.0), 1.0)
    return "#{:02x}{:02x}{:02x}".format(
        round(r1 * (1 - ratio) + r2 * ratio),
        round(g1 * (1 - ratio) + g2 * ratio),
        round(b1 * (1 - ratio) + b2 * ratio),
    )


def build_fig07_fig08_outputs(
    results_dir: str | Path,
    figures_dir: str | Path,
    events_path: str | Path | None = None,
    slope_raster_path: str | Path | None = None,
    slope_all_touched: bool = False,
    mapping_threshold: float = 0.80,
) -> dict[str, object]:
    '''
    构建 Fig.7/Fig.8 源数据与 HTML 预览图。

    :param results_dir: DID 结果目录。
    :param figures_dir: HTML 图表输出目录。
    :param events_path: 当前滑坡事件矢量路径。
    :param slope_raster_path: 坡度栅格路径。
    :param slope_all_touched: 是否统计所有被滑坡边界触碰到的坡度像元。
    :param mapping_threshold: 生成当前滑坡清单所用的概率阈值。
    :return: 数据表、已写文件和图表路径组成的字典。
    '''
    results_dir = ensure_dir(results_dir)
    figures_dir = ensure_dir(figures_dir)
    source_root = ensure_dir(Path(results_dir).parent / "06_figure_source_data")
    tables = build_recovery_summary_tables(
        results_dir,
        events_path=events_path,
        slope_raster_path=slope_raster_path,
        slope_all_touched=slope_all_touched,
    )
    first_year_valid = tables["fig08_first_year_recovery_metrics"][
        tables["fig08_first_year_recovery_metrics"]["valid_first_year_recovery"].astype(bool)
    ].copy()
    fig07_source = tables["fig07_mean_size_trajectory"].merge(
        tables["fig07_mean_fit_size_summary"],
        on=["outcome", "size_class"],
        how="left",
        validate="many_to_one",
    )
    fig07_source.insert(0, "mapping_threshold", float(mapping_threshold))

    area_summary = tables["fig08_first_year_recovery_area_bin_summary"].drop(
        columns=["bin_type", "bin_order"], errors="ignore"
    ).rename(columns={"bin_label": "area_bin"})
    area_summary = area_summary.rename(
        columns={col: f"area_group_{col}" for col in area_summary.columns if col not in {"outcome", "area_bin"}}
    )
    slope_summary = tables["fig08_first_year_recovery_slope_bin_summary"].drop(
        columns=["bin_type", "bin_order"], errors="ignore"
    ).rename(columns={"bin_label": "slope_bin"})
    slope_summary = slope_summary.rename(
        columns={col: f"slope_group_{col}" for col in slope_summary.columns if col not in {"outcome", "slope_bin"}}
    )
    fig08_source = first_year_valid.merge(
        area_summary,
        on=["outcome", "area_bin"],
        how="left",
        validate="many_to_one",
    ).merge(
        slope_summary,
        on=["outcome", "slope_bin"],
        how="left",
        validate="many_to_one",
    )
    fig08_source.insert(0, "mapping_threshold", float(mapping_threshold))

    fig08_summary = pd.concat(
        [
            tables["fig08_first_year_recovery_area_bin_summary"],
            tables["fig08_first_year_recovery_slope_bin_summary"],
        ],
        ignore_index=True,
    )
    fig08_summary.insert(0, "mapping_threshold", float(mapping_threshold))
    fig08_summary = fig08_summary.sort_values(
        ["outcome", "bin_type", "bin_order", "bin_label"],
        kind="stable",
    ).reset_index(drop=True)

    written_tables = [
        _write_figure_source_table(
            fig07_source,
            source_root / "Fig07_Recovery" / "Fig07_Recovery_Trajectories.csv",
        ),
        _write_figure_source_table(
            fig08_source,
            source_root / "Fig08_FirstYear_Recovery" / "Fig08_FirstYear_Recovery.csv",
        ),
        _write_figure_source_table(
            fig08_summary,
            source_root / "Fig08_FirstYear_Recovery" / "Fig08_FirstYear_Recovery_Summary.csv",
        ),
    ]
    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_fig08_first_year_recovery_boxplots(
            metrics=tables["fig08_first_year_recovery_metrics"],
            area_summary=tables["fig08_first_year_recovery_area_bin_summary"],
            slope_summary=tables["fig08_first_year_recovery_slope_bin_summary"],
            figures_dir=figures_dir,
        )
    )
    figure_paths.extend(
        plot_fig07_cumulative_size_bars(
            size_cum=tables["fig07_mean_size_trajectory"],
            figures_dir=figures_dir,
        )
    )
    return {
        "tables": tables,
        "figure_source_tables": {
            "Fig07": fig07_source,
            "Fig08": fig08_source,
            "Fig08_summary": fig08_summary,
        },
        "written_tables": written_tables,
        "figures": figure_paths,
    }
