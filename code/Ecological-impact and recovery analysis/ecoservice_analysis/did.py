from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .io_utils import ensure_dir, safe_to_csv


def _normal_inference(att: float, se: float | None, alpha: float) -> dict[str, float | None]:
    """
    根据 ATT 和标准误计算正态近似推断结果。

    :param att: 平均处理效应估计值。Average Treatment Effect on the Treated
    :param se: 标准误。
    :param alpha: 显著性水平，例如 0.05 对应 95% 置信区间。
    :return: 标准误、z 值、p 值和置信区间上下界。
    """
    if se is None or not np.isfinite(se) or se <= 0:
        return {"se": None, "z": None, "p_value": None, "ci_low": None, "ci_high": None}
    z = att / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "se": float(se),
        "z": float(z),
        "p_value": float(p),
        "ci_low": float(att - crit * se),
        "ci_high": float(att + crit * se),
    }


def _append_event_study_reference_rows(
    event_study: pd.DataFrame,
    reference_relative_year: int = -1,
) -> pd.DataFrame:
    """Add explicit zero-effect rows for the event-study reference period."""
    if event_study.empty or "relative_year" not in event_study.columns:
        return event_study
    group_cols = [col for col in ["outcome", "zone_name"] if col in event_study.columns]
    if not group_cols:
        return event_study

    df = event_study.copy()
    df["relative_year"] = pd.to_numeric(df["relative_year"], errors="coerce")
    ref = int(reference_relative_year)
    rows: list[dict[str, Any]] = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if (chunk["relative_year"] == ref).any():
            continue
        base = chunk.sort_values("relative_year").iloc[0].to_dict()
        for col, key in zip(group_cols, keys):
            base[col] = key
        base["relative_year"] = ref
        for col in ["att", "se", "z", "ci_low", "ci_high"]:
            if col in base:
                base[col] = 0.0
        if "p_value" in base:
            base["p_value"] = 1.0
        for col in ["n_cohorts", "n_treated", "n_control_mean"]:
            if col in base:
                base[col] = np.nan
        rows.append(base)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df.sort_values(group_cols + ["relative_year"]).reset_index(drop=True)


def _unit_changes(
    wide: pd.DataFrame,
    unit_ids: pd.Index,
    year: int,
    baseline_year: int,
) -> pd.Series:
    """
    计算一组空间单元从基准年到目标年的变化量。

    :param wide: 行为空间单元、列为年份的宽表。
    :param unit_ids: 需要计算的空间单元编号。
    :param year: 目标年份。
    :param baseline_year: DID 基准年份，通常为滑坡前一年。
    :return: 每个空间单元的变化量序列。
    """
    if year not in wide.columns or baseline_year not in wide.columns:
        return pd.Series(dtype="float64")
    available = unit_ids.intersection(wide.index)
    if len(available) == 0:
        return pd.Series(dtype="float64")
    changes = wide.loc[available, year] - wide.loc[available, baseline_year]
    return changes.dropna()


def _area_weights(unit_info: pd.DataFrame, changes: pd.Series) -> pd.Series:
    """
    Return positive area weights aligned to a unit-change series.

    Each row in the DID panel is a landslide zone. For spatial summaries, a 100 m2
    landslide should contribute ten times as much as a 10 m2 landslide when their
    loss percentages are aggregated.
    """
    if changes.empty or "area_m2" not in unit_info.columns:
        return pd.Series(1.0, index=changes.index, dtype="float64")
    weights = pd.to_numeric(unit_info.reindex(changes.index)["area_m2"], errors="coerce")
    weights = weights.where(weights > 0)
    if weights.notna().sum() == 0:
        return pd.Series(1.0, index=changes.index, dtype="float64")
    return weights.fillna(weights.dropna().median()).astype(float)


def _weighted_mean_var_n(values: pd.Series, weights: pd.Series) -> tuple[float, float, float]:
    """Compute area-weighted mean, weighted variance, and effective sample size."""
    aligned = pd.concat([values.rename("value"), weights.rename("weight")], axis=1).dropna()
    aligned = aligned[aligned["weight"] > 0]
    if aligned.empty:
        return np.nan, np.nan, 0.0
    x = aligned["value"].astype(float).to_numpy()
    w = aligned["weight"].astype(float).to_numpy()
    w_sum = float(w.sum())
    mean = float(np.average(x, weights=w))
    var = float(np.average((x - mean) ** 2, weights=w)) if len(x) > 1 else 0.0
    n_eff = float((w_sum**2) / np.sum(w**2)) if np.sum(w**2) > 0 else float(len(x))
    return mean, var, n_eff


def _pre_baseline_deltas(
    wide: pd.DataFrame,
    unit_ids: pd.Index,
    cohort_year: int,
    reference_relative_year: int,
    pre_relative_years: list[int],
    min_years: int,
) -> pd.Series:
    """
    计算每个空间单元的灾前变化均值，用来筛选与 core 平行趋势更接近的 never-control。

    这里不用灾后 ATT 做匹配，而是只用 t-1 之前的年份：
    delta = mean(Y_pre_year - Y_reference_year)。
    """
    baseline_year = int(cohort_year + reference_relative_year)
    if baseline_year not in wide.columns:
        return pd.Series(dtype="float64")
    years = [
        int(cohort_year + rel_year)
        for rel_year in pre_relative_years
        if int(rel_year) < int(reference_relative_year) and int(cohort_year + rel_year) in wide.columns
    ]
    if len(years) < int(min_years):
        return pd.Series(dtype="float64")
    available = unit_ids.intersection(wide.index)
    if len(available) == 0:
        return pd.Series(dtype="float64")

    baseline = wide.loc[available, baseline_year]
    changes = wide.loc[available, years].sub(baseline, axis=0)
    valid_counts = changes.notna().sum(axis=1)
    deltas = changes.mean(axis=1)
    return deltas[(valid_counts >= int(min_years)) & deltas.notna()]


def _build_never_control_match_map(
    unit_info: pd.DataFrame,
    wide: pd.DataFrame,
    cohort_years: list[int],
    reference_relative_year: int,
    pre_relative_years: list[int],
    threshold: float,
    min_years: int,
    source_zone: str,
    empty_action: str = "none",
    nearest_fraction: float = 0.05,
    nearest_min_units: int = 1,
) -> tuple[dict[int, pd.Index], list[dict[str, Any]]]:
    """
    为每个 cohort 生成灾前趋势相似的 never-control 单元池。

    判断标准为：abs(候选 never-control 灾前变化均值 - core 灾前变化均值) <= threshold。
    """
    match_map: dict[int, pd.Index] = {}
    diagnostics: list[dict[str, Any]] = []
    never_units = unit_info.index[unit_info["zone_role"] == "never_control"]

    for cohort_year in cohort_years:
        source_units = unit_info.index[
            (unit_info["zone_role"] == "treated")
            & (unit_info["zone_name"] == source_zone)
            & (unit_info["event_year"] == cohort_year)
        ]
        source_deltas = _pre_baseline_deltas(
            wide=wide,
            unit_ids=source_units,
            cohort_year=int(cohort_year),
            reference_relative_year=reference_relative_year,
            pre_relative_years=pre_relative_years,
            min_years=min_years,
        )
        candidate_deltas = _pre_baseline_deltas(
            wide=wide,
            unit_ids=never_units,
            cohort_year=int(cohort_year),
            reference_relative_year=reference_relative_year,
            pre_relative_years=pre_relative_years,
            min_years=min_years,
        )

        match_method = "empty"
        min_candidate_distance = np.nan
        matched_distance_mean = np.nan
        if source_deltas.empty or candidate_deltas.empty:
            matched_units = pd.Index([])
            source_mean = np.nan
        else:
            source_mean = float(source_deltas.mean())
            distance = (candidate_deltas - source_mean).abs()
            min_candidate_distance = float(distance.min()) if not distance.empty else np.nan
            matched_distance = distance[distance <= float(threshold)].sort_values()
            if matched_distance.empty and str(empty_action).lower() == "nearest":
                fallback_n = min(
                    len(distance),
                    max(
                        int(nearest_min_units),
                        int(np.ceil(len(distance) * max(float(nearest_fraction), 0.0))),
                    ),
                )
                matched_distance = distance.sort_values().head(fallback_n)
                match_method = "nearest_fallback"
            else:
                match_method = "threshold"
            matched_units = pd.Index(matched_distance.index)
            if not matched_distance.empty:
                matched_distance_mean = float(matched_distance.mean())

        match_map[int(cohort_year)] = matched_units
        diagnostics.append(
            {
                "cohort_year": int(cohort_year),
                "source_zone": source_zone,
                "threshold": float(threshold),
                "min_pre_years": int(min_years),
                "source_delta_mean": source_mean,
                "n_source_units": int(len(source_deltas)),
                "n_candidate_never_controls": int(len(candidate_deltas)),
                "n_matched_never_controls": int(len(matched_units)),
                "match_method": match_method,
                "min_candidate_distance": min_candidate_distance,
                "matched_distance_mean": matched_distance_mean,
            }
        )

    return match_map, diagnostics


def _estimate_group_time_att(
    data: pd.DataFrame,
    unit_info: pd.DataFrame,
    wide: pd.DataFrame,
    target_zone: str,
    cohort_year: int,
    year: int,
    reference_relative_year: int,
    control_group: str,
    min_treated_units: int,
    min_control_units: int,
    alpha: float,
    matched_never_control_units: pd.Index | None = None,
) -> dict[str, Any] | None:
    """
    估计某个滑坡发生年份组在某个年份的组-时间 ATT。

    :param data: 当前 outcome 与当前 zone 的长表数据。
    :param unit_info: 每个空间单元的属性表。
    :param wide: 空间单元 x 年份的宽表。
    :param target_zone: 当前估计的区域类型，例如 core 或 ring_0_20m。
    :param cohort_year: 处理组滑坡发生年份。
    :param year: 目标观测年份。
    :param reference_relative_year: 基准相对年份，默认 -1。
    :param control_group: 控制组策略，支持 not_yet、never、not_yet_plus_never。
    :param min_treated_units: 处理组最少空间单元数。
    :param min_control_units: 控制组最少空间单元数。
    :param alpha: 显著性水平。
    :return: 当前组-时间 ATT 结果；样本不足时返回 None。
    """
    baseline_year = int(cohort_year + reference_relative_year)
    if year == baseline_year:
        return None

    treated_units = unit_info.index[
        (unit_info["zone_role"] == "treated")
        & (unit_info["zone_name"] == target_zone)
        & (unit_info["event_year"] == cohort_year)
    ]
    treated_changes = _unit_changes(wide, treated_units, year, baseline_year)
    if len(treated_changes) < min_treated_units:
        return None
    treated_weights = _area_weights(unit_info, treated_changes)

    control_filters = []
    max_control_year = max(int(year), int(baseline_year))
    if control_group in {"not_yet", "not_yet_plus_never"}:
        control_filters.append(
            (unit_info["zone_role"] == "treated")
            & (unit_info["zone_name"] == target_zone)
            & (unit_info["event_year"] > max_control_year)
            & (unit_info["event_year"] != cohort_year)
        )
    if control_group in {"never", "not_yet_plus_never"}:
        never_filter = unit_info["zone_role"] == "never_control"
        if matched_never_control_units is not None:
            never_filter = never_filter & unit_info.index.isin(matched_never_control_units)
        control_filters.append(never_filter)
    if not control_filters:
        raise ValueError("control_group must be not_yet, never, or not_yet_plus_never")

    control_filter = control_filters[0].copy()
    for extra_filter in control_filters[1:]:
        control_filter = control_filter | extra_filter
    control_units = unit_info.index[control_filter]
    control_changes = _unit_changes(wide, control_units, year, baseline_year)
    if len(control_changes) < min_control_units:
        return None
    control_weights = _area_weights(unit_info, control_changes)

    treated_mean, treated_var, treated_n_eff = _weighted_mean_var_n(treated_changes, treated_weights)
    control_mean, control_var, control_n_eff = _weighted_mean_var_n(control_changes, control_weights)
    if not np.isfinite(treated_mean) or not np.isfinite(control_mean) or treated_n_eff <= 0 or control_n_eff <= 0:
        return None
    att = treated_mean - control_mean
    se = float(np.sqrt(treated_var / treated_n_eff + control_var / control_n_eff))
    inference = _normal_inference(att, se, alpha)

    zone_rows = unit_info[
        (unit_info["zone_role"] == "treated") & (unit_info["zone_name"] == target_zone)
    ]
    distance_min = float(zone_rows["distance_min_m"].iloc[0])
    distance_max = float(zone_rows["distance_max_m"].iloc[0])

    return {
        "zone_name": target_zone,
        "cohort_year": int(cohort_year),
        "year": int(year),
        "baseline_year": int(baseline_year),
        "relative_year": int(year - cohort_year),
        "att": float(att),
        "treated_change": treated_mean,
        "control_change": control_mean,
        "n_treated": int(len(treated_changes)),
        "n_control": int(len(control_changes)),
        "treated_area_m2": float(treated_weights.reindex(treated_changes.index).sum()),
        "control_area_m2": float(control_weights.reindex(control_changes.index).sum()),
        "treated_effective_n": float(treated_n_eff),
        "control_effective_n": float(control_n_eff),
        "distance_min_m": distance_min,
        "distance_max_m": distance_max,
        **inference,
    }


def _aggregate_event_study(
    
    group_time: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    把组-时间 ATT 按相对年份聚合成事件研究曲线。

    :param group_time: 每个 cohort-year 的 ATT 表。
    :param alpha: 显著性水平。
    :return: 按 outcome、zone、relative_year 汇总后的事件研究表。
    """
    rows: list[dict[str, Any]] = []
    if group_time.empty:
        return pd.DataFrame()

    group_cols = ["outcome", "zone_name", "relative_year"]
    for keys, chunk in group_time.groupby(group_cols):
        outcome, zone_name, rel_year = keys
        if "treated_area_m2" in chunk.columns:
            weights = pd.to_numeric(chunk["treated_area_m2"], errors="coerce").fillna(0).astype(float).to_numpy()
        else:
            weights = chunk["n_treated"].astype(float).to_numpy()
        if weights.sum() == 0:
            continue
        att_values = chunk["att"].astype(float).to_numpy()
        att = float(np.average(att_values, weights=weights))
        ses = chunk["se"].astype(float).to_numpy()
        valid_se = np.isfinite(ses)
        se = None
        if np.any(valid_se):
            norm_w = weights[valid_se] / weights[valid_se].sum()
            se = float(np.sqrt(np.sum((norm_w**2) * (ses[valid_se] ** 2))))
        inference = _normal_inference(att, se, alpha)
        rows.append(
            {
                "outcome": outcome,
                "zone_name": zone_name,
                "relative_year": int(rel_year),
                "att": att,
                "n_cohorts": int(chunk["cohort_year"].nunique()),
                "n_treated": int(chunk["n_treated"].sum()),
                "treated_area_m2": float(np.sum(weights)),
                "n_treated_weight_mode": "area_m2" if "treated_area_m2" in chunk.columns else "unit_count",
                "n_control_mean": float(chunk["n_control"].mean()),
                "control_area_m2_mean": float(pd.to_numeric(chunk.get("control_area_m2"), errors="coerce").mean())
                if "control_area_m2" in chunk.columns
                else np.nan,
                "distance_min_m": float(chunk["distance_min_m"].iloc[0]),
                "distance_max_m": float(chunk["distance_max_m"].iloc[0]),
                **inference,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["outcome", "distance_min_m", "distance_max_m", "relative_year"])
    return result


def estimate_staggered_did(
    
    panel: pd.DataFrame,
    results_dir: str | Path,
    value_column: str = "mean",
    reference_relative_year: int = -1,
    event_window: list[int] | tuple[int, int] = (-5, 6),
    control_group: str = "not_yet_plus_never",
    min_treated_units: int = 3,
    min_control_units: int = 3,
    alpha: float = 0.05,
    pretrend_match_never_control: bool = False,
    pretrend_match_threshold: float = 0.001,
    pretrend_match_source_zone: str = "core",
    pretrend_match_min_years: int = 1,
    pretrend_match_empty_action: str = "none",
    pretrend_match_nearest_fraction: float = 0.05,
    pretrend_match_nearest_min_units: int = 1,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    估计适合交错滑坡事件的组-时间 DID。

    :param panel: 面板数据，每行是一个空间单元在某年的生态变量统计值。
    :param results_dir: DID 结果输出目录。
    :param value_column: 使用哪个统计值作为结果变量，默认 mean。
    :param reference_relative_year: 基准相对年份，默认滑坡前一年。
    :param event_window: 事件研究窗口，例如 [-5, 5]。
    :param control_group: 控制组策略。
    :param min_treated_units: 处理组最少样本数。
    :param min_control_units: 控制组最少样本数。
    :param alpha: 显著性水平。
    :param logger: 日志对象。
    :return: 组-时间 ATT 表和聚合后的事件研究 ATT 表。
    """
    results_dir = ensure_dir(results_dir)
    df = panel.copy()
    df["event_year"] = pd.to_numeric(df["event_year"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=[value_column])

    if isinstance(event_window, (list, tuple)) and len(event_window) == 2:
        rel_years = list(range(int(event_window[0]), int(event_window[1]) + 1))
    else:
        rel_years = [int(x) for x in event_window]
    pre_relative_years = [k for k in rel_years if k < int(reference_relative_year)]
    rel_years = [k for k in rel_years if k != reference_relative_year]

    group_time_rows: list[pd.DataFrame] = []
    match_diagnostics: list[dict[str, Any]] = []
    for outcome, outcome_df in df.groupby("outcome"):
        outcome_unit_info = (
            outcome_df.sort_values(["unit_id", "year"])
            .drop_duplicates("unit_id")
            .set_index("unit_id")
        )
        outcome_wide = outcome_df.pivot_table(
            index="unit_id",
            columns="year",
            values=value_column,
            aggfunc="mean",
        )
        treated_zone_names = sorted(
            outcome_df.loc[outcome_df["zone_role"] == "treated", "zone_name"].dropna().unique().tolist()
        )
        outcome_cohorts = sorted(
            outcome_unit_info.loc[
                (outcome_unit_info["zone_role"] == "treated")
                & outcome_unit_info["event_year"].notna(),
                "event_year",
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        never_match_map: dict[int, pd.Index] = {}
        if pretrend_match_never_control:
            never_match_map, diagnostics = _build_never_control_match_map(
                unit_info=outcome_unit_info,
                wide=outcome_wide,
                cohort_years=outcome_cohorts,
                reference_relative_year=reference_relative_year,
                pre_relative_years=pre_relative_years,
                threshold=pretrend_match_threshold,
                min_years=pretrend_match_min_years,
                source_zone=pretrend_match_source_zone,
                empty_action=pretrend_match_empty_action,
                nearest_fraction=pretrend_match_nearest_fraction,
                nearest_min_units=pretrend_match_nearest_min_units,
            )
            for item in diagnostics:
                item["outcome"] = outcome
            match_diagnostics.extend(diagnostics)
            if logger is not None:
                matched_total = sum(int(item["n_matched_never_controls"]) for item in diagnostics)
                logger.info(
                    "Pretrend-matched never controls for %s: %s matched units across %s cohorts",
                    outcome,
                    matched_total,
                    len(diagnostics),
                )
        for target_zone in treated_zone_names:
            target_df = outcome_df[
                (
                    (outcome_df["zone_role"] == "treated")
                    & (outcome_df["zone_name"] == target_zone)
                )
                | (outcome_df["zone_role"] == "never_control")
            ].copy()
            if target_df.empty:
                continue
            unit_info = (
                target_df.sort_values(["unit_id", "year"])
                .drop_duplicates("unit_id")
                .set_index("unit_id")
            )
            wide = target_df.pivot_table(
                index="unit_id",
                columns="year",
                values=value_column,
                aggfunc="mean",
            )
            cohorts = sorted(
                unit_info.loc[
                    (unit_info["zone_role"] == "treated")
                    & (unit_info["zone_name"] == target_zone)
                    & unit_info["event_year"].notna(),
                    "event_year",
                ]
                .astype(int)
                .unique()
                .tolist()
            )
            years = sorted(target_df["year"].dropna().astype(int).unique().tolist())
            rows: list[dict[str, Any]] = []
            for cohort_year in cohorts:
                for rel_year in rel_years:
                    year = int(cohort_year + rel_year)
                    if year not in years:
                        continue
                    row = _estimate_group_time_att(
                        data=target_df,
                        unit_info=unit_info,
                        wide=wide,
                        target_zone=target_zone,
                        cohort_year=int(cohort_year),
                        year=year,
                        reference_relative_year=reference_relative_year,
                        control_group=control_group,
                        min_treated_units=min_treated_units,
                        min_control_units=min_control_units,
                        alpha=alpha,
                        matched_never_control_units=(
                            never_match_map.get(int(cohort_year)) if pretrend_match_never_control else None
                        ),
                    )
                    if row is not None:
                        row["outcome"] = outcome
                        rows.append(row)
            if rows:
                zone_gt = pd.DataFrame(rows)
                group_time_rows.append(zone_gt)
                if logger is not None:
                    logger.info(
                        "Estimated %s group-time ATT rows for %s / %s",
                        len(zone_gt),
                        outcome,
                        target_zone,
                    )

    group_time = pd.concat(group_time_rows, ignore_index=True) if group_time_rows else pd.DataFrame()
    event_study = _aggregate_event_study(group_time, alpha=alpha)
    event_study = _append_event_study_reference_rows(event_study, reference_relative_year)

    group_time_path = results_dir / "group_time_att.csv"
    event_study_path = results_dir / "event_study_att.csv"
    match_diagnostics_path = results_dir / "never_control_pretrend_match.csv"
    group_time.to_csv(group_time_path, index=False, encoding="utf-8-sig")
    event_study.to_csv(event_study_path, index=False, encoding="utf-8-sig")
    if match_diagnostics:
        match_diagnostics_path = safe_to_csv(
            pd.DataFrame(match_diagnostics),
            match_diagnostics_path,
            logger=logger,
            index=False,
            encoding="utf-8-sig",
        )
    if logger is not None:
        logger.info("Wrote group-time ATT to %s", group_time_path)
        logger.info("Wrote event-study ATT to %s", event_study_path)
        if match_diagnostics:
            logger.info("Wrote never-control pretrend diagnostics to %s", match_diagnostics_path)
    return group_time, event_study
