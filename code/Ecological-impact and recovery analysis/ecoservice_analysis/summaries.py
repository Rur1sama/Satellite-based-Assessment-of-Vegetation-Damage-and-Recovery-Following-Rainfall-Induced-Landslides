from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, safe_to_csv


def _append_relative_reference_rows(
    df: pd.DataFrame,
    group_cols: list[str],
    reference_relative_year: int = -1,
    zero_cols: list[str] | None = None,
    one_cols: list[str] | None = None,
    nan_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add explicit zero rows for the relative-year reference period."""
    if df.empty or "relative_year" not in df.columns:
        return df
    group_cols = [col for col in group_cols if col in df.columns]
    if not group_cols:
        return df

    result = df.copy()
    result["relative_year"] = pd.to_numeric(result["relative_year"], errors="coerce")
    ref = int(reference_relative_year)
    zero_cols = zero_cols or []
    one_cols = one_cols or []
    nan_cols = nan_cols or []
    rows: list[dict[str, Any]] = []

    for keys, chunk in result.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if (chunk["relative_year"] == ref).any():
            continue
        base = chunk.sort_values("relative_year").iloc[0].to_dict()
        for col, key in zip(group_cols, keys):
            base[col] = key
        base["relative_year"] = ref
        if "cohort_year" in base and pd.notna(base["cohort_year"]):
            cohort_year = int(float(base["cohort_year"]))
            if "year" in base:
                base["year"] = cohort_year + ref
            if "baseline_year" in base:
                base["baseline_year"] = cohort_year + ref
        for col in zero_cols:
            if col in base:
                base[col] = 0.0
        for col in one_cols:
            if col in base:
                base[col] = 1.0
        for col in nan_cols:
            if col in base:
                base[col] = np.nan
        rows.append(base)

    if rows:
        result = pd.concat([result, pd.DataFrame(rows)], ignore_index=True)
    return result.sort_values(group_cols + ["relative_year"]).reset_index(drop=True)


def _append_att_reference_rows(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
    reference_relative_year: int = -1,
) -> pd.DataFrame:
    """Add t=-1 zero ATT rows for event-study style tables."""
    group_cols = group_cols or ["outcome", "zone_name"]
    return _append_relative_reference_rows(
        df,
        group_cols=group_cols,
        reference_relative_year=reference_relative_year,
        zero_cols=["att", "se", "z", "ci_low", "ci_high", "treated_change", "control_change"],
        one_cols=["p_value"],
        nan_cols=["n_cohorts", "n_treated", "n_control", "n_control_mean"],
    )


def summarize_radiation_range(
    event_study: pd.DataFrame,
    output_dir: str | Path,
    post_years: list[int],
    alpha: float = 0.05,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    if event_study.empty:
        result = pd.DataFrame()
        safe_to_csv(result, output_dir / "radiation_range_summary.csv", index=False, encoding="utf-8-sig")
        return result

    df = event_study.copy()
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    df["att"] = pd.to_numeric(df["att"], errors="coerce")

    for outcome in sorted(df["outcome"].dropna().unique()):
        for rel_year in post_years:
            chunk = df[
                (df["outcome"] == outcome)
                & (df["relative_year"] == rel_year)
                & (df["distance_max_m"] > 0)
            ].copy()
            sig = chunk[(chunk["att"] < 0) & (chunk["p_value"] <= alpha)]
            rows.append(
                {
                    "outcome": outcome,
                    "relative_year": int(rel_year),
                    "n_significant_negative_zones": int(len(sig)),
                    "max_significant_distance_m": float(sig["distance_max_m"].max())
                    if not sig.empty
                    else None,
                    "significant_zones": ";".join(sig["zone_name"].astype(str).tolist()),
                }
            )

        post_chunk = df[
            (df["outcome"] == outcome)
            & (df["relative_year"].isin(post_years))
            & (df["distance_max_m"] > 0)
        ].copy()
        if not post_chunk.empty:
            zone_summary = (
                post_chunk.groupby(["zone_name", "distance_min_m", "distance_max_m"])
                .agg(
                    post_mean_att=("att", "mean"),
                    n_sig_negative=("p_value", lambda s: int(((post_chunk.loc[s.index, "att"] < 0) & (s <= alpha)).sum())),
                )
                .reset_index()
            )
            safe_to_csv(
                zone_summary,
                output_dir / f"radiation_zone_post_mean_{outcome}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    result = pd.DataFrame(rows)
    safe_to_csv(result, output_dir / "radiation_range_summary.csv", index=False, encoding="utf-8-sig")
    return result


def _treated_counterfactual_from_att(
    att_table: pd.DataFrame,
    panel: pd.DataFrame,
    value_column: str = "mean",
) -> pd.DataFrame:
    """Attach treated observed means and DID counterfactual means to an ATT table."""
    if att_table.empty or panel.empty:
        return pd.DataFrame()

    att = att_table.copy()
    att["relative_year"] = pd.to_numeric(att["relative_year"], errors="coerce")
    if att.empty:
        return pd.DataFrame()

    p = panel.copy()
    if "zone_role" in p.columns:
        p = p[p["zone_role"].astype(str).eq("treated")].copy()
    p["relative_year"] = pd.to_numeric(p["relative_year"], errors="coerce")
    p[value_column] = pd.to_numeric(p[value_column], errors="coerce")
    p["area_m2"] = pd.to_numeric(p["area_m2"], errors="coerce")
    if "event_year" in p.columns:
        p["cohort_year"] = pd.to_numeric(p["event_year"], errors="coerce")

    keys = ["outcome", "zone_name", "relative_year"]
    if "cohort_year" in att.columns and "cohort_year" in p.columns:
        keys.append("cohort_year")

    p = p.dropna(subset=keys + [value_column, "area_m2"]).copy()
    p = p[p["area_m2"] > 0].copy()
    if p.empty:
        return att.assign(
            observed_treated_mean=np.nan,
            total_area_m2=np.nan,
            counterfactual_mean=np.nan,
            counterfactual_total=np.nan,
        )

    p["weighted_value"] = p[value_column] * p["area_m2"]
    observed = (
        p.groupby(keys, as_index=False)
        .agg(weighted_value=("weighted_value", "sum"), total_area_m2=("area_m2", "sum"))
    )
    observed["observed_treated_mean"] = observed["weighted_value"] / observed["total_area_m2"]
    observed = observed.drop(columns=["weighted_value"])

    result = att.merge(observed, on=keys, how="left")
    result["att"] = pd.to_numeric(result["att"], errors="coerce")
    result["counterfactual_mean"] = result["observed_treated_mean"] - result["att"]
    result["counterfactual_total"] = result["counterfactual_mean"] * result["total_area_m2"]
    return result


def summarize_percent_event_study(
    event_study: pd.DataFrame,
    panel: pd.DataFrame,
    output_dir: str | Path,
    value_column: str = "mean",
) -> pd.DataFrame:
    """
    把 DID ATT 转换为相对滑坡前处理区均值的百分比效应。
    """
    output_dir = ensure_dir(output_dir)
    if event_study.empty or panel.empty:
        result = pd.DataFrame()
        safe_to_csv(result, output_dir / "event_study_att_percent.csv", index=False, encoding="utf-8-sig")
        return result

    event_study = _append_att_reference_rows(event_study)
    result = _treated_counterfactual_from_att(event_study, panel, value_column=value_column)
    valid = result["counterfactual_mean"].notna() & (result["counterfactual_mean"] > 0)
    for src, dst in [("att", "att_percent"), ("ci_low", "ci_low_percent"), ("ci_high", "ci_high_percent")]:
        result[dst] = np.nan
        result.loc[valid, dst] = result.loc[valid, src] / result.loc[valid, "counterfactual_mean"] * 100.0
    safe_to_csv(result, output_dir / "event_study_att_percent.csv", index=False, encoding="utf-8-sig")
    return result

