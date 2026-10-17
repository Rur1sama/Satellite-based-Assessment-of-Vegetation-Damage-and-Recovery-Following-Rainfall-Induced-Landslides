from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .did import estimate_staggered_did
from .io_utils import ensure_dir, safe_to_csv
from .summaries import summarize_percent_event_study


def _normalise_event_id(panel: pd.DataFrame) -> pd.DataFrame:
    """确保 panel 中存在 event_id；旧表缺列时从 unit_id 里拆出来。"""
    result = panel.copy()
    if "event_id" not in result.columns:
        result["event_id"] = result["unit_id"].astype(str).str.split("__").str[0]
    return result


def _size_class_order(label: str) -> int:
    """让 small 始终排在 large 前面，其他自定义标签排在后面。"""
    text = str(label).lower()
    if text in {"small", "no_inner_small"}:
        return 0
    if text in {"large", "inner_available_large"}:
        return 1
    return 99


def add_weighted_core_zone(
    panel: pd.DataFrame,
    core_zone_name: str = "core",
    inner_zone_name: str = "core_inner",
    weighted_zone_name: str = "core_weighted",
    inner_weight: float = 0.8,
    edge_weight: float = 0.2,
    value_column: str = "mean",
) -> pd.DataFrame:
    """
    Add a derived direct zone: weighted core = inner_weight * inner core + edge_weight * edge.

    The edge mean is reconstructed from existing area-weighted means:
    edge_mean = (core_mean * core_area - inner_mean * inner_area) / (core_area - inner_area).
    If the edge cannot be separated, the derived row falls back to the original core mean.
    """
    if panel.empty or value_column not in panel.columns:
        return panel

    df = _normalise_event_id(panel)
    required = {"outcome", "year", "event_id", "zone_role", "zone_name", "area_m2", value_column}
    if not required.issubset(df.columns):
        return panel

    core = df[(df["zone_role"] == "treated") & (df["zone_name"] == core_zone_name)].copy()
    inner = df[(df["zone_role"] == "treated") & (df["zone_name"] == inner_zone_name)].copy()
    if core.empty:
        return panel

    key_cols = [
        col
        for col in ["outcome", "year", "event_id", "event_year", "matched_event_year", "relative_year"]
        if col in df.columns
    ]
    if not key_cols:
        return panel

    core = core.drop_duplicates(key_cols).copy()
    inner_cols = key_cols + [value_column, "area_m2"]
    if "n_pixels" in inner.columns:
        inner_cols.append("n_pixels")
    inner = inner.drop_duplicates(key_cols)[inner_cols].copy() if not inner.empty else pd.DataFrame(columns=inner_cols)
    merged = core.merge(inner, on=key_cols, how="left", suffixes=("", "_inner"))

    iw = float(inner_weight)
    ew = float(edge_weight)
    total_w = iw + ew
    if total_w <= 0:
        iw, ew = 0.8, 0.2
    else:
        iw, ew = iw / total_w, ew / total_w

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        core_value = pd.to_numeric(pd.Series([row.get(value_column)]), errors="coerce").iloc[0]
        core_area = pd.to_numeric(pd.Series([row.get("area_m2")]), errors="coerce").iloc[0]
        inner_value = pd.to_numeric(pd.Series([row.get(f"{value_column}_inner")]), errors="coerce").iloc[0]
        inner_area = pd.to_numeric(pd.Series([row.get("area_m2_inner")]), errors="coerce").iloc[0]

        weighted_value = core_value
        method = "core_fallback"
        edge_area = pd.NA
        edge_value = pd.NA
        if pd.notna(core_value) and pd.notna(core_area) and float(core_area) > 0:
            if pd.notna(inner_value) and pd.notna(inner_area) and float(inner_area) > 0:
                edge_area_float = float(core_area) - float(inner_area)
                if edge_area_float > 1e-9:
                    edge_value_float = (
                        float(core_value) * float(core_area) - float(inner_value) * float(inner_area)
                    ) / edge_area_float
                    weighted_value = iw * float(inner_value) + ew * edge_value_float
                    edge_area = edge_area_float
                    edge_value = edge_value_float
                    method = f"inner{iw:.2f}_edge{ew:.2f}"
                else:
                    weighted_value = float(core_value)
                    edge_area = 0.0
                    method = "no_separable_edge"

        new_row = row[core.columns].to_dict()
        event_id = str(new_row.get("event_id", "event"))
        new_row["unit_id"] = f"{event_id}__{weighted_zone_name}"
        new_row["zone_id"] = f"{event_id}__{weighted_zone_name}"
        new_row["zone_name"] = weighted_zone_name
        new_row[value_column] = weighted_value
        if "median" in new_row:
            new_row["median"] = weighted_value
        if "std" in new_row:
            new_row["std"] = pd.NA
        if "min" in new_row:
            new_row["min"] = weighted_value
        if "max" in new_row:
            new_row["max"] = weighted_value
        new_row["weighted_core_inner_weight"] = iw
        new_row["weighted_core_edge_weight"] = ew
        new_row["weighted_core_method"] = method
        new_row["weighted_core_edge_area_m2"] = edge_area
        new_row["weighted_core_edge_mean"] = edge_value
        rows.append(new_row)

    if not rows:
        return panel

    weighted = pd.DataFrame(rows).dropna(axis=1, how="all")
    result = df[df["zone_name"] != weighted_zone_name].copy()
    result = result.dropna(axis=1, how="all")
    return pd.concat([result, weighted], ignore_index=True, sort=False)


def build_core_size_class_panel(
    panel: pd.DataFrame,
    threshold_m2: float | None = None,
    quantile: float = 0.5,
    core_zone_name: str = "core",
    class_method: str = "area_threshold",
    inner_zone_name: str = "core_inner",
    small_label: str = "small",
    large_label: str = "large",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按滑坡 core 面积给事件分成 small / large，并只保留 core 与对应 never-control。

    :param panel: 已提取好的 zone-outcome 面板。
    :param threshold_m2: 面积阈值；None 时使用 core 面积分位数。
    :param quantile: threshold_m2 为空时使用的分位数，默认中位数。
    :param core_zone_name: 坡体核心区的 zone_name。
    :param small_label: 小滑坡标签。
    :param large_label: 大滑坡标签。
    :return: size_panel、事件分组表、分组统计表。
    """
    if panel.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = _normalise_event_id(panel)
    required = {"event_id", "zone_role", "zone_name", "area_m2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing required columns for core size comparison: {sorted(missing)}")

    core_events = (
        df[(df["zone_role"] == "treated") & (df["zone_name"] == core_zone_name)]
        .drop_duplicates("event_id")
        [["event_id", "event_year", "area_m2"]]
        .copy()
    )
    core_events["area_m2"] = pd.to_numeric(core_events["area_m2"], errors="coerce")
    core_events = core_events.dropna(subset=["area_m2"])
    if core_events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    class_method = str(class_method or "area_threshold")
    if class_method == "inner_availability":
        # 几何分组：能形成 inner core 的斑块通常更大，内部像元更少受边缘混合影响。
        inner_events = set(
            df[(df["zone_role"] == "treated") & (df["zone_name"] == inner_zone_name)]["event_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        core_events["size_class"] = core_events["event_id"].astype(str).map(
            lambda event_id: large_label if event_id in inner_events else small_label
        )
        threshold = float("nan")
        threshold_source = f"inner_availability_{inner_zone_name}"
    else:
        if threshold_m2 is None:
            threshold = float(core_events["area_m2"].quantile(float(quantile)))
            threshold_source = f"quantile_{float(quantile):.3g}"
        else:
            threshold = float(threshold_m2)
            threshold_source = "config_threshold_m2"

        # 使用 < / >= 保证 1000 m2 这类边界值进入大滑坡组。
        core_events["size_class"] = core_events["area_m2"].where(
            core_events["area_m2"] < threshold,
            other=pd.NA,
        )
        core_events["size_class"] = core_events["size_class"].map(
            lambda value: small_label if pd.notna(value) else large_label
        )
    core_events["size_threshold_m2"] = threshold
    core_events["threshold_source"] = threshold_source
    core_events["size_zone_name"] = core_zone_name

    # 若阈值碰到极端分布导致某一组为空，则按排序强制二分，避免后续没有对比图。
    counts = core_events["size_class"].value_counts()
    if len(counts) < 2 and class_method != "inner_availability":
        ranked = core_events.sort_values(["area_m2", "event_id"]).reset_index(drop=True)
        half = max(len(ranked) // 2, 1)
        ranked["size_class"] = [small_label if idx < half else large_label for idx in ranked.index]
        ranked["threshold_source"] = f"rank_split_quantile_{float(quantile):.3g}"
        core_events = ranked

    summary = (
        core_events.groupby("size_class", as_index=False)
        .agg(
            n_events=("event_id", "count"),
            total_core_area_m2=("area_m2", "sum"),
            mean_core_area_m2=("area_m2", "mean"),
            median_core_area_m2=("area_m2", "median"),
            min_core_area_m2=("area_m2", "min"),
            max_core_area_m2=("area_m2", "max"),
            size_threshold_m2=("size_threshold_m2", "first"),
            threshold_source=("threshold_source", "first"),
            size_zone_name=("size_zone_name", "first"),
        )
        .sort_values("size_class", key=lambda s: s.map(_size_class_order))
    )

    class_map = core_events[
        ["event_id", "size_class", "area_m2", "size_threshold_m2", "threshold_source", "size_zone_name"]
    ].rename(
        columns={"area_m2": "landslide_core_area_m2"}
    )
    size_panel = df.merge(class_map, on="event_id", how="left")
    size_panel = size_panel.dropna(subset=["size_class"]).copy()

    # 只看坡体 core 的处理效应；never-control 仅作为反事实参考，不作为损失空间。
    keep = (
        ((size_panel["zone_role"] == "treated") & (size_panel["zone_name"] == core_zone_name))
        | (size_panel["zone_role"] == "never_control")
    )
    size_panel = size_panel[keep].copy()
    return size_panel, class_map.sort_values("event_id"), summary

def _log_core_size_summary(summary: pd.DataFrame, logger: logging.Logger | None = None) -> None:
    """把大/小滑坡分组结果写入日志，便于每次调整阈值后快速核对。"""
    if summary.empty or logger is None:
        return
    threshold = pd.to_numeric(summary["size_threshold_m2"], errors="coerce").dropna()
    threshold_text = f"{float(threshold.iloc[0]):.0f} m2" if not threshold.empty else "unknown"
    parts: list[str] = []
    ordered = summary.sort_values("size_class", key=lambda s: s.map(_size_class_order))
    for _, row in ordered.iterrows():
        parts.append(
            (
                f"{row['size_class']}={int(row['n_events'])} events "
                f"(area {float(row['min_core_area_m2']):.0f}-"
                f"{float(row['max_core_area_m2']):.0f} m2)"
            )
        )
    zone_text = str(summary["size_zone_name"].dropna().iloc[0]) if "size_zone_name" in summary.columns else "core"
    logger.info("Size split zone=%s, threshold=%s: %s", zone_text, threshold_text, "; ".join(parts))


def run_core_size_class_analysis(
    panel: pd.DataFrame,
    results_dir: str | Path,
    outcome_cfgs: dict[str, dict[str, Any]],
    did_cfg: dict[str, Any],
    size_cfg: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    output_prefix: str = "core_size",
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    分别对 small / large 滑坡 core 区重新估计 DID，并输出用于绘图的合并表。

    返回表包括：
    - core_size_group_time_att.csv
    - core_size_event_study_att.csv
    - core_size_event_study_att_percent.csv
    - core_size_class_assignment.csv
    - core_size_class_summary.csv
    """
    results_dir = ensure_dir(results_dir)
    size_cfg = size_cfg or {}
    output_prefix = str(size_cfg.get("output_prefix", output_prefix) or "core_size")
    value_column = str(did_cfg.get("value_column", "mean"))
    weighted_zone_name = str(size_cfg.get("weighted_zone_name", "core_weighted"))
    zone_names_raw = size_cfg.get("zone_names")
    requested_zone_names: list[str] = []
    if zone_names_raw is not None:
        if isinstance(zone_names_raw, str):
            requested_zone_names = [zone_names_raw]
        else:
            requested_zone_names = [str(zone) for zone in zone_names_raw if str(zone).strip()]
    elif size_cfg.get("zone_name") is not None:
        requested_zone_names = [str(size_cfg.get("zone_name"))]
    if weighted_zone_name in requested_zone_names:
        panel = add_weighted_core_zone(
            panel=panel,
            core_zone_name=str(size_cfg.get("weighted_core_source_zone", "core")),
            inner_zone_name=str(size_cfg.get("weighted_core_inner_zone", "core_inner")),
            weighted_zone_name=weighted_zone_name,
            inner_weight=float(size_cfg.get("weighted_core_inner_weight", 0.8)),
            edge_weight=float(size_cfg.get("weighted_core_edge_weight", 0.2)),
            value_column=value_column,
        )
    if zone_names_raw is not None:
        if isinstance(zone_names_raw, str):
            zone_names = [zone_names_raw]
        else:
            zone_names = [str(zone) for zone in zone_names_raw if str(zone).strip()]
        zone_names = list(dict.fromkeys(zone_names))
        if len(zone_names) > 1:
            collected: dict[str, list[pd.DataFrame]] = {
                "assignment": [],
                "summary": [],
                "group_time": [],
                "event_study": [],
                "event_study_percent": [],
            }
            panel_tables: list[pd.DataFrame] = []
            for zone_name in zone_names:
                single_cfg = dict(size_cfg)
                single_cfg.pop("zone_names", None)
                single_cfg["zone_name"] = zone_name
                zone_dir = Path(tempfile.mkdtemp(prefix=f"{output_prefix}_{zone_name}_"))
                outputs = run_core_size_class_analysis(
                    panel=panel,
                    results_dir=zone_dir,
                    outcome_cfgs=outcome_cfgs,
                    did_cfg={**did_cfg, "pretrend_match_source_zone": zone_name},
                    size_cfg=single_cfg,
                    logger=logger,
                    output_prefix=output_prefix,
                    write_outputs=False,
                )
                if not outputs.get("panel", pd.DataFrame()).empty:
                    panel_tables.append(outputs["panel"].assign(size_zone_name=zone_name))
                for key in collected:
                    table = outputs.get(key, pd.DataFrame())
                    if table is not None and not table.empty:
                        collected[key].append(table.assign(size_zone_name=zone_name))

            combined = {
                key: pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
                for key, tables in collected.items()
            }
            combined["panel"] = pd.concat(panel_tables, ignore_index=True) if panel_tables else pd.DataFrame()
            if write_outputs:
                safe_to_csv(combined["assignment"], results_dir / f"{output_prefix}_class_assignment.csv", index=False, encoding="utf-8-sig")
                safe_to_csv(combined["summary"], results_dir / f"{output_prefix}_class_summary.csv", index=False, encoding="utf-8-sig")
                safe_to_csv(combined["group_time"], results_dir / f"{output_prefix}_group_time_att.csv", index=False, encoding="utf-8-sig")
                safe_to_csv(
                    combined["event_study"],
                    results_dir / f"{output_prefix}_event_study_att.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                safe_to_csv(
                    combined["event_study_percent"],
                    results_dir / f"{output_prefix}_event_study_att_percent.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            return combined

    threshold_m2 = size_cfg.get("area_threshold_m2")
    core_zone_name = str(size_cfg.get("zone_name", "core"))
    # 手动面积阈值优先；只有阈值为空时，才读取 area_quantile 做自动分位数分组。
    quantile = float(size_cfg.get("area_quantile", 0.5)) if threshold_m2 is None else 0.5
    size_panel, assignment, summary = build_core_size_class_panel(
        panel=panel,
        threshold_m2=threshold_m2,
        quantile=quantile,
        core_zone_name=core_zone_name,
        class_method=str(size_cfg.get("class_method", "area_threshold")),
        inner_zone_name=str(size_cfg.get("inner_zone_name", "core_inner")),
        small_label=str(size_cfg.get("small_label", "small")),
        large_label=str(size_cfg.get("large_label", "large")),
    )
    if write_outputs:
        safe_to_csv(assignment, results_dir / f"{output_prefix}_class_assignment.csv", index=False, encoding="utf-8-sig")
        safe_to_csv(summary, results_dir / f"{output_prefix}_class_summary.csv", index=False, encoding="utf-8-sig")
    _log_core_size_summary(summary, logger=logger)

    if size_panel.empty or summary.empty:
        empty = pd.DataFrame()
        empty_names = [
            "core_size_group_time_att.csv",
            "core_size_event_study_att.csv",
            "core_size_event_study_att_percent.csv",
        ]
        for name in empty_names:
            if write_outputs:
                safe_to_csv(empty, results_dir / name.replace("core_size", output_prefix), index=False, encoding="utf-8-sig")
        return {
            "panel": size_panel,
            "assignment": assignment,
            "summary": summary,
            "group_time": empty,
            "event_study": empty,
            "event_study_percent": empty,
        }

    if write_outputs:
        tmp_root = ensure_dir(results_dir / f"_{output_prefix}_class_tmp")
    else:
        # 递归计算只需要工作目录；使用随机临时目录避免旧进程锁住固定文件名。
        tmp_root = Path(tempfile.mkdtemp(prefix=f"{output_prefix}_did_"))
    group_tables: list[pd.DataFrame] = []
    event_tables: list[pd.DataFrame] = []
    event_percent_tables: list[pd.DataFrame] = []

    labels = sorted(size_panel["size_class"].dropna().unique().tolist(), key=_size_class_order)
    for label in labels:
        label_panel = size_panel[size_panel["size_class"] == label].copy()
        tmp_dir = ensure_dir(tmp_root / str(label)) if write_outputs else tmp_root
        if logger is not None:
            n_events = label_panel.loc[label_panel["zone_name"].eq(core_zone_name), "event_id"].nunique()
            logger.info("Estimating %s-only DID for %s landslides (%s events)", core_zone_name, label, n_events)

        group_time, event_study = estimate_staggered_did(
            panel=label_panel,
            results_dir=tmp_dir,
            value_column=did_cfg.get("value_column", "mean"),
            reference_relative_year=int(did_cfg.get("reference_relative_year", -1)),
            event_window=did_cfg.get("event_window", [-5, 5]),
            control_group=did_cfg.get("control_group", "not_yet_plus_never"),
            min_treated_units=int(did_cfg.get("min_treated_units", 1)),
            min_control_units=int(did_cfg.get("min_control_units", 1)),
            alpha=float(did_cfg.get("alpha", 0.05)),
            pretrend_match_never_control=bool(did_cfg.get("pretrend_match_never_control", False)),
            pretrend_match_threshold=float(did_cfg.get("pretrend_match_threshold", 0.001)),
            pretrend_match_source_zone=str(did_cfg.get("pretrend_match_source_zone", "core")),
            pretrend_match_min_years=int(did_cfg.get("pretrend_match_min_years", 1)),
            pretrend_match_empty_action=str(did_cfg.get("pretrend_match_empty_action", "none")),
            pretrend_match_nearest_fraction=float(did_cfg.get("pretrend_match_nearest_fraction", 0.05)),
            pretrend_match_nearest_min_units=int(did_cfg.get("pretrend_match_nearest_min_units", 1)),
            logger=logger,
        )
        if not group_time.empty:
            group_time = group_time.assign(size_class=label, size_zone_name=core_zone_name)
            group_tables.append(group_time)
        if not event_study.empty:
            event_study = event_study.assign(size_class=label, size_zone_name=core_zone_name)
            event_tables.append(event_study)

            event_percent = summarize_percent_event_study(
                event_study=event_study,
                panel=label_panel,
                output_dir=tmp_dir,
                value_column=did_cfg.get("value_column", "mean"),
            ).assign(size_class=label, size_zone_name=core_zone_name)
            event_percent_tables.append(event_percent)


    combined = {
        "group_time": pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame(),
        "event_study": pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame(),
        "event_study_percent": (
            pd.concat(event_percent_tables, ignore_index=True) if event_percent_tables else pd.DataFrame()
        ),
    }

    if write_outputs:
        safe_to_csv(combined["group_time"], results_dir / f"{output_prefix}_group_time_att.csv", index=False, encoding="utf-8-sig")
        safe_to_csv(
            combined["event_study"],
            results_dir / f"{output_prefix}_event_study_att.csv",
            index=False,
            encoding="utf-8-sig",
        )
        safe_to_csv(
            combined["event_study_percent"],
            results_dir / f"{output_prefix}_event_study_att_percent.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # 分组 DID 的临时目录只服务于本次汇总；最终结果写出后立即清理，避免 04_did_results 长期堆积。
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    return {
        "panel": size_panel,
        "assignment": assignment,
        "summary": summary,
        **combined,
    }

