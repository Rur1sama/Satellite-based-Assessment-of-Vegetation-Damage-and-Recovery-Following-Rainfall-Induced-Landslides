from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ecoservice_analysis.area_response import run_area_size_response_analysis
from ecoservice_analysis.core_size import add_weighted_core_zone, run_core_size_class_analysis
from ecoservice_analysis.did import estimate_staggered_did
from ecoservice_analysis.events import masks_to_landslide_events
from ecoservice_analysis.individual_diagnostics import run_individual_drop_diagnostics
from ecoservice_analysis.io_utils import (
    ensure_dir,
    load_config,
    mask_specs_from_config,
    outcome_specs_from_config,
    resolve_path,
    setup_logger,
)
from ecoservice_analysis.panel import extract_zone_outcome_panel
from ecoservice_analysis.plotting import (
    plot_event_study_percent,
    plot_radiation_profiles,
)
from ecoservice_analysis.recovery_figures import build_fig07_fig08_outputs
from ecoservice_analysis.summaries import (
    summarize_percent_event_study,
    summarize_radiation_range,
)
from ecoservice_analysis.zones import build_landslide_zones


def _parse_steps(raw_steps: list[str]) -> list[str]:
    if not raw_steps or "all" in raw_steps:
        return ["events", "zones", "panel", "did", "plots"]
    valid = {"events", "zones", "panel", "did", "plots"}
    unknown = sorted(set(raw_steps) - valid)
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}")
    return raw_steps


def _preferred_csv(path: Path) -> Path:
    """返回标准结果文件路径；保留函数是为了集中管理 CSV 读取入口。"""
    return path


def _configured_treated_zones(cfg: dict) -> list[str] | None:
    """读取主分析要保留的 treated zone；None 表示保留全部。"""
    zones = cfg.get("analysis_scope", {}).get("treated_zones")
    if zones is None:
        return None
    if isinstance(zones, str):
        zones = [zones]
    zones = [str(zone) for zone in zones if str(zone).strip()]
    return zones or None


def _configured_enabled_outcomes(cfg: dict) -> list[str]:
    """读取当前主分析启用的 outcome，旧缓存里的禁用指标会被过滤掉。"""
    outcomes = cfg.get("outcomes", {})
    enabled: list[str] = []
    for name, spec in outcomes.items():
        if bool(spec.get("enabled", False)):
            enabled.append(str(name))
    return enabled


def _filter_panel_outcomes(panel: pd.DataFrame | None, enabled_outcomes: list[str]) -> pd.DataFrame | None:
    """只保留当前配置启用的生态指标，避免旧 panel 缓存把 FAPAR 等中间变量带回主结果。"""
    if panel is None or panel.empty or not enabled_outcomes or "outcome" not in panel.columns:
        return panel
    return panel[panel["outcome"].astype(str).isin(enabled_outcomes)].copy()


def _filter_panel_treated_zones(panel: pd.DataFrame | None, treated_zones: list[str] | None) -> pd.DataFrame | None:
    """只保留指定 treated zone 和 never-control；用于只看坡体 core 的分析。"""
    if panel is None or panel.empty or not treated_zones or "zone_name" not in panel.columns:
        return panel
    if "zone_role" not in panel.columns:
        return panel[panel["zone_name"].isin(treated_zones)].copy()
    keep = (panel["zone_role"] != "treated") | panel["zone_name"].isin(treated_zones)
    return panel[keep].copy()


def _filter_result_treated_zones(df: pd.DataFrame | None, treated_zones: list[str] | None) -> pd.DataFrame | None:
    """过滤已有 DID 结果表，避免 plots 步读取旧的 ring 结果。"""
    if df is None or df.empty or not treated_zones:
        return df
    result = df.copy()
    if "zone_name" in result.columns:
        result = result[result["zone_name"].isin(treated_zones)].copy()
    if "loss_type" in result.columns and set(treated_zones) == {"core"}:
        result = result[result["loss_type"] == "Direct"].copy()
    return result


def _filter_result_outcomes(df: pd.DataFrame | None, enabled_outcomes: list[str]) -> pd.DataFrame | None:
    """过滤已保存的 DID 结果表，保持图表和当前 outcome 配置一致。"""
    if df is None or df.empty or not enabled_outcomes or "outcome" not in df.columns:
        return df
    return df[df["outcome"].astype(str).isin(enabled_outcomes)].copy()


def _configured_weighted_core_zone(cfg: dict) -> str | None:
    """Return weighted-core zone name when any configured output asks for it."""
    analysis_zones = _configured_treated_zones(cfg) or []
    size_cfg = cfg.get("core_size_comparison", {})
    zone_names = size_cfg.get("zone_names", [])
    if isinstance(zone_names, str):
        zone_names = [zone_names]
    weighted_name = str(size_cfg.get("weighted_zone_name", "core_weighted"))
    wanted = set(str(zone) for zone in analysis_zones) | set(str(zone) for zone in zone_names)
    return weighted_name if weighted_name in wanted else None


def _add_configured_weighted_core_zone(
    panel: pd.DataFrame | None,
    cfg: dict,
    did_cfg: dict,
    logger,
) -> pd.DataFrame | None:
    """Add the derived weighted-core rows used by DID plots without re-extracting rasters."""
    weighted_name = _configured_weighted_core_zone(cfg)
    if panel is None or panel.empty or weighted_name is None:
        return panel
    size_cfg = cfg.get("core_size_comparison", {})
    before = len(panel)
    result = add_weighted_core_zone(
        panel=panel,
        core_zone_name=str(size_cfg.get("weighted_core_source_zone", "core")),
        inner_zone_name=str(size_cfg.get("weighted_core_inner_zone", "core_inner")),
        weighted_zone_name=weighted_name,
        inner_weight=float(size_cfg.get("weighted_core_inner_weight", 0.8)),
        edge_weight=float(size_cfg.get("weighted_core_edge_weight", 0.2)),
        value_column=str(did_cfg.get("value_column", "mean")),
    )
    added = len(result) - before
    if added > 0 and logger is not None:
        logger.info("Added %s derived weighted-core panel rows as %s", added, weighted_name)
    return result


def _required_panel_unit_ids(zones_path: Path, treated_zones: list[str] | None) -> set[str]:
    """读取当前评价口径真正需要的 zone_id；never-control 始终保留为 DID 反事实对照。"""
    zones = gpd.read_file(zones_path)
    if zones.empty or "zone_id" not in zones.columns:
        return set()
    if treated_zones and "zone_role" in zones.columns and "zone_name" in zones.columns:
        keep = (zones["zone_role"] != "treated") | zones["zone_name"].astype(str).isin(treated_zones)
        zones = zones[keep].copy()
    return set(zones["zone_id"].astype(str).tolist())


def _required_panel_zone_meta(zones_path: Path, treated_zones: list[str] | None) -> pd.DataFrame:
    """读取当前 zone 元数据，用于判断 panel 缓存是否仍对应同一批空间单元。"""
    zones = gpd.read_file(zones_path)
    if zones.empty or "zone_id" not in zones.columns:
        return pd.DataFrame()
    if treated_zones and "zone_role" in zones.columns and "zone_name" in zones.columns:
        keep = (zones["zone_role"] != "treated") | zones["zone_name"].astype(str).isin(treated_zones)
        zones = zones[keep].copy()
    cols = [
        col
        for col in ["zone_id", "event_id", "event_year", "zone_name", "zone_role", "area_m2"]
        if col in zones.columns
    ]
    meta = zones[cols].copy()
    meta["unit_id"] = meta["zone_id"].astype(str)
    for col in ["event_id", "zone_name", "zone_role"]:
        if col in meta.columns:
            meta[col] = meta[col].astype(str)
    for col in ["event_year", "area_m2"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce").round(6)
    return meta.drop_duplicates("unit_id")


def _panel_cache_matches_zones(cached: pd.DataFrame, zone_meta: pd.DataFrame, logger) -> bool:
    """确认缓存 panel 的 unit_id 没有继承旧事件编号或旧面积。"""
    if cached.empty or zone_meta.empty or "unit_id" not in cached.columns:
        return False
    compare_cols = [
        col
        for col in ["event_id", "event_year", "zone_name", "zone_role", "area_m2"]
        if col in cached.columns and col in zone_meta.columns
    ]
    if not compare_cols:
        return False
    cached_meta = cached[["unit_id", *compare_cols]].drop_duplicates("unit_id").copy()
    cached_meta["unit_id"] = cached_meta["unit_id"].astype(str)
    for col in ["event_id", "zone_name", "zone_role"]:
        if col in cached_meta.columns:
            cached_meta[col] = cached_meta[col].astype(str)
    for col in ["event_year", "area_m2"]:
        if col in cached_meta.columns:
            cached_meta[col] = pd.to_numeric(cached_meta[col], errors="coerce").round(6)
    current = zone_meta[["unit_id", *compare_cols]].copy()
    merged = cached_meta.merge(current, on="unit_id", how="inner", suffixes=("_cached", "_current"))
    if len(merged) != len(cached_meta):
        return False
    mismatch = pd.Series(False, index=merged.index)
    for col in compare_cols:
        mismatch = mismatch | (merged[f"{col}_cached"] != merged[f"{col}_current"])
    n_bad = int(mismatch.sum())
    if n_bad:
        if logger is not None:
            example_cols = ["unit_id"]
            for col in compare_cols:
                example_cols.extend([f"{col}_cached", f"{col}_current"])
            examples = merged.loc[mismatch, example_cols].head(5)
            logger.warning(
                "Panel cache is stale: %s unit metadata rows differ from current zones. Examples:\n%s",
                n_bad,
                examples.to_string(index=False),
            )
        return False
    return True


def _try_incremental_panel_extract(
    zones_path: Path,
    panel_path: Path,
    outcome_specs: dict,
    treated_zones: list[str] | None,
    all_touched: bool,
    min_valid_pixels: int,
    cache_dir: Path | None,
    logger,
) -> pd.DataFrame | None:
    """增量补齐 panel 中缺失的 zone / outcome / year，避免新增 GPP 后误用旧缓存。"""
    if panel_path is None or not Path(panel_path).exists():
        return None
    try:
        cached = pd.read_csv(panel_path)
    except Exception:
        return None
    if cached.empty or "unit_id" not in cached.columns:
        return None

    required_ids = _required_panel_unit_ids(Path(zones_path), treated_zones)
    if not required_ids:
        return None

    cached["unit_id"] = cached["unit_id"].astype(str)
    cached_keep = cached[cached["unit_id"].isin(required_ids)].copy()
    zone_meta = _required_panel_zone_meta(Path(zones_path), treated_zones)
    if not _panel_cache_matches_zones(cached_keep, zone_meta, logger):
        if logger is not None:
            logger.warning("Discard panel cache and recompute all zones because cached unit metadata is stale.")
        return None
    existing_ids = set(cached_keep["unit_id"].dropna().astype(str).unique().tolist())

    # 旧逻辑只检查 unit_id，新增 outcome 时会误判为“不缺数据”。
    # 这里把两类缺口拆开：
    # 1) 新增 zone：只给这些新 zone 补所有 outcome/year；
    # 2) 新增 outcome/year：只给这些 outcome/year 补全部既有 zone。
    missing_specs: dict[str, dict] = {}
    missing_id_union: set[str] = set()
    if cached_keep.empty:
        missing_specs = outcome_specs
        missing_id_union = set(required_ids)
    else:
        cached_key_counts = (
            cached_keep.groupby(["outcome", "year"])["unit_id"].nunique().to_dict()
            if {"outcome", "year", "unit_id"}.issubset(cached_keep.columns)
            else {}
        )
        for outcome_name, spec in outcome_specs.items():
            missing_years: dict[int, list[Path]] = {}
            for year, raster_paths in spec.get("rasters", {}).items():
                present_count = int(cached_key_counts.get((outcome_name, int(year)), 0))
                # 对已经有面板记录的 unit 判断 outcome/year 是否完整；
                # 极少数无有效像元的 zone 会单独走 missing_ids 分支，不应触发全量 outcome 重提取。
                if present_count < len(existing_ids):
                    missing_years[int(year)] = raster_paths
                    missing_id_union.update(existing_ids or required_ids)
            if missing_years:
                missing_spec = dict(spec)
                missing_spec["rasters"] = missing_years
                missing_specs[outcome_name] = missing_spec

    missing_ids = required_ids - existing_ids
    if logger is not None:
        logger.info(
            "Panel incremental check: required_units=%s, cached_units=%s, missing_units=%s, missing_outcomes=%s",
            len(required_ids),
            len(existing_ids),
            len(missing_ids),
            sorted(missing_specs.keys()),
        )
    if not missing_specs:
        if not missing_ids:
            cached_keep.to_csv(panel_path, index=False, encoding="utf-8-sig")
            return cached_keep

    missing_panels: list[pd.DataFrame] = []
    if missing_ids:
        missing_zone_path = panel_path.with_name(f"{panel_path.stem}_missing_units_tmp.csv")
        missing_panels.append(
            extract_zone_outcome_panel(
                zones_path=zones_path,
                outcome_specs=outcome_specs,
                output_csv=missing_zone_path,
                all_touched=all_touched,
                min_valid_pixels=min_valid_pixels,
                unit_ids=missing_ids,
                cache_dir=cache_dir,
                logger=logger,
            )
        )
    if missing_specs:
        missing_outcome_path = panel_path.with_name(f"{panel_path.stem}_missing_outcomes_tmp.csv")
        missing_panels.append(
            extract_zone_outcome_panel(
                zones_path=zones_path,
                outcome_specs=missing_specs,
                output_csv=missing_outcome_path,
                all_touched=all_touched,
                min_valid_pixels=min_valid_pixels,
                unit_ids=missing_id_union or required_ids,
                cache_dir=cache_dir,
                logger=logger,
            )
        )

    panel = pd.concat([cached_keep, *missing_panels], ignore_index=True, sort=False)
    dedupe_cols = [col for col in ["outcome", "year", "unit_id"] if col in panel.columns]
    if dedupe_cols:
        panel = panel.drop_duplicates(dedupe_cols, keep="last")
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    if logger is not None:
        logger.info("Wrote incremental panel with %s rows to %s", len(panel), panel_path)
    return panel


def run_pipeline(config_path: str | Path, steps: list[str], force: bool = False) -> None:
    '''
    运行滑坡生态服务 DID 全流程。

    :param config_path: JSON 配置文件路径。
    :param steps: 需要执行的步骤，可选 events、zones、panel、did、plots 或 all。
    :param force: 是否强制重算已有结果；为 False 时会优先复用已经落盘的中间结果。
    :return: None
    '''
    cfg, config_dir = load_config(config_path)
    output_dir = ensure_dir(resolve_path(cfg["output_dir"], config_dir))
    logger = setup_logger(output_dir / "ecoservice_did_pipeline.log")
    logger.info("Loaded config: %s", config_path)

    steps = _parse_steps(steps)
    logger.info("Running steps: %s", ", ".join(steps))

    landslide_cfg = cfg["landslide"]
    zone_cfg = cfg["zones"]
    panel_cfg = cfg["panel"]
    did_cfg = cfg["did"]
    plot_cfg = cfg.get("plots", {})
    recovery_cfg = cfg.get("recovery", {})
    treated_zones = _configured_treated_zones(cfg)
    enabled_outcomes = _configured_enabled_outcomes(cfg)
    skip_radiation_profiles = bool(
        cfg.get("analysis_scope", {}).get("skip_radiation_profiles", bool(treated_zones))
    )

    events_path = resolve_path(landslide_cfg["events_output"], config_dir)
    zones_path = resolve_path(zone_cfg["zones_output"], config_dir)
    panel_path = resolve_path(panel_cfg["panel_output"], config_dir)
    default_panel_cache_dir = Path(panel_path).with_name(f"{Path(panel_path).stem}_outcome_year_cache")
    panel_cache_dir = resolve_path(panel_cfg.get("extraction_cache_dir", default_panel_cache_dir), config_dir)
    study_area_path = resolve_path(cfg.get("study_area_path"), config_dir)
    did_dir = ensure_dir(resolve_path(did_cfg["results_dir"], config_dir))
    figures_dir = ensure_dir(resolve_path(plot_cfg.get("figures_dir", output_dir / "05_figures"), config_dir))
    slope_raster_path = resolve_path(
        recovery_cfg.get("slope_raster_path", output_dir.parent / "Slope.tif"),
        config_dir,
    )

    if "events" in steps and events_path is not None and Path(events_path).exists() and not force:
        logger.info("Use cached landslide events: %s", events_path)
    elif "events" in steps:
        mask_specs = mask_specs_from_config(cfg, config_dir, logger)
        masks_to_landslide_events(
            mask_specs=mask_specs,
            output_path=events_path,
            analysis_crs=cfg.get("analysis_crs"),
            study_area_path=study_area_path,
            mask_threshold=float(landslide_cfg.get("mask_threshold", 0.8)),
            min_area_m2=float(landslide_cfg.get("min_landslide_area_m2", 0.0)),
            connectivity=int(landslide_cfg.get("connectivity", 8)),
            logger=logger,
        )

    if "zones" in steps and zones_path is not None and Path(zones_path).exists() and not force:
        logger.info("Use cached analysis zones: %s", zones_path)
    elif "zones" in steps:
        if events_path is None or not Path(events_path).exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        build_landslide_zones(
            events_path=events_path,
            output_path=zones_path,
            spillover_rings_m=zone_cfg.get("spillover_rings_m", []),
            study_area_path=study_area_path,
            make_core_inner_zone=bool(zone_cfg.get("make_core_inner_zone", False)),
            core_inner_erosion_m=float(zone_cfg.get("core_inner_erosion_m", 5.0)),
            core_inner_zone_name=str(zone_cfg.get("core_inner_zone_name", "core_inner")),
            make_never_control_zones=bool(zone_cfg.get("make_never_control_zones", True)),
            never_control_ring_m=zone_cfg.get("never_control_ring_m"),
            exclude_all_landslides=bool(zone_cfg.get("exclude_all_landslide_polygons_from_rings", True)),
            min_zone_area_m2=float(zone_cfg.get("min_zone_area_m2", 0.0)),
            logger=logger,
        )

    panel = None
    outcome_specs = outcome_specs_from_config(cfg, config_dir, logger)
    use_panel_cache = bool(panel_cfg.get("use_cache", True))
    if (
        "panel" in steps
        and use_panel_cache
        and panel_path is not None
        and Path(panel_path).exists()
        and not force
    ):
        logger.info("Use cached extracted panel: %s", panel_path)
        panel = pd.read_csv(panel_path)
    elif "panel" in steps:
        if zones_path is None or not Path(zones_path).exists():
            raise FileNotFoundError(f"Zones file not found: {zones_path}")
        if force and bool(panel_cfg.get("incremental_on_force", False)):
            panel = _try_incremental_panel_extract(
                zones_path=Path(zones_path),
                panel_path=Path(panel_path),
                outcome_specs=outcome_specs,
                treated_zones=treated_zones,
                all_touched=bool(panel_cfg.get("all_touched", False)),
                min_valid_pixels=int(panel_cfg.get("min_valid_pixels", 1)),
                cache_dir=Path(panel_cache_dir),
                logger=logger,
            )
        if panel is None:
            panel = extract_zone_outcome_panel(
                zones_path=zones_path,
                outcome_specs=outcome_specs,
                output_csv=panel_path,
                all_touched=bool(panel_cfg.get("all_touched", False)),
                min_valid_pixels=int(panel_cfg.get("min_valid_pixels", 1)),
                cache_dir=panel_cache_dir,
                reuse_cache=not force,
                logger=logger,
            )
    elif panel_path is not None and Path(panel_path).exists():
        panel = pd.read_csv(panel_path)
    panel = _filter_panel_outcomes(panel, enabled_outcomes)
    panel = _filter_panel_treated_zones(panel, treated_zones)
    panel = _add_configured_weighted_core_zone(panel, cfg, did_cfg, logger)

    group_time = None
    event_study = None
    percent_event_study = None
    core_size_outputs = None
    core_size_cfg = cfg.get("core_size_comparison", {})
    run_core_size = bool(core_size_cfg.get("enabled", False))
    if "did" in steps:
        if panel is None:
            if panel_path is None or not Path(panel_path).exists():
                raise FileNotFoundError(f"Panel file not found: {panel_path}")
            panel = pd.read_csv(panel_path)
            panel = _filter_panel_outcomes(panel, enabled_outcomes)
            panel = _filter_panel_treated_zones(panel, treated_zones)
            panel = _add_configured_weighted_core_zone(panel, cfg, did_cfg, logger)
        group_time, event_study = estimate_staggered_did(
            panel=panel,
            results_dir=did_dir,
            value_column=did_cfg.get("value_column", "mean"),
            reference_relative_year=int(did_cfg.get("reference_relative_year", -1)),
            event_window=did_cfg.get("event_window", [-5, 6]),
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
        summarize_radiation_range(
            event_study=event_study,
            output_dir=did_dir,
            post_years=[int(x) for x in did_cfg.get("radiation_post_years", [0, 1, 2])],
            alpha=float(did_cfg.get("alpha", 0.05)),
        )
        percent_event_study = summarize_percent_event_study(
            event_study=event_study,
            panel=panel,
            output_dir=did_dir,
            value_column=did_cfg.get("value_column", "mean"),
        )
        if run_core_size:
            core_size_outputs = run_core_size_class_analysis(
                panel=panel,
                results_dir=did_dir,
                outcome_cfgs=cfg.get("outcomes", {}),
                did_cfg=did_cfg,
                size_cfg=core_size_cfg,
                logger=logger,
            )
    elif (did_dir / "event_study_att.csv").exists():
        group_time_path = _preferred_csv(did_dir / "group_time_att.csv")
        if group_time_path.exists():
            group_time = pd.read_csv(group_time_path)
        event_study = pd.read_csv(_preferred_csv(did_dir / "event_study_att.csv"))
        group_time = _filter_result_outcomes(group_time, enabled_outcomes)
        event_study = _filter_result_outcomes(event_study, enabled_outcomes)
        group_time = _filter_result_treated_zones(group_time, treated_zones)
        event_study = _filter_result_treated_zones(event_study, treated_zones)
        percent_event_study_path = _preferred_csv(did_dir / "event_study_att_percent.csv")
        if percent_event_study_path.exists():
            percent_event_study = _filter_result_outcomes(pd.read_csv(percent_event_study_path), enabled_outcomes)
            percent_event_study = _filter_result_treated_zones(percent_event_study, treated_zones)

    if "plots" in steps:
        if event_study is None:
            event_study_path = _preferred_csv(did_dir / "event_study_att.csv")
            if not event_study_path.exists():
                raise FileNotFoundError(f"Event-study file not found: {event_study_path}")
            event_study = pd.read_csv(event_study_path)
            event_study = _filter_result_outcomes(event_study, enabled_outcomes)
            event_study = _filter_result_treated_zones(event_study, treated_zones)
        if group_time is None:
            group_time_path = _preferred_csv(did_dir / "group_time_att.csv")
            group_time = pd.read_csv(group_time_path) if group_time_path.exists() else pd.DataFrame()
            group_time = _filter_result_outcomes(group_time, enabled_outcomes)
            group_time = _filter_result_treated_zones(group_time, treated_zones)
        if percent_event_study is None:
            percent_event_study_path = _preferred_csv(did_dir / "event_study_att_percent.csv")
            percent_event_study = (
                pd.read_csv(percent_event_study_path) if percent_event_study_path.exists() else pd.DataFrame()
            )
            percent_event_study = _filter_result_outcomes(percent_event_study, enabled_outcomes)
            percent_event_study = _filter_result_treated_zones(percent_event_study, treated_zones)
        if panel is not None and not panel.empty:
            percent_event_study = summarize_percent_event_study(
                event_study=event_study,
                panel=panel,
                output_dir=did_dir,
                value_column=did_cfg.get("value_column", "mean"),
            )
            template_rasters_by_year = {
                int(spec["event_year"]): spec["path"]
                for spec in mask_specs_from_config(cfg, config_dir, logger=None)
            }
            run_individual_drop_diagnostics(
                panel=panel,
                zones_path=zones_path,
                output_dir=did_dir,
                figures_dir=figures_dir,
                value_column=did_cfg.get("value_column", "mean"),
                template_rasters_by_year=template_rasters_by_year,
                study_area_path=study_area_path,
                logger=logger,
            )
        if bool(plot_cfg.get("legacy_dashboards", False)):
            plot_event_study_percent(percent_event_study, figures_dir)
            if not skip_radiation_profiles:
                plot_radiation_profiles(
                    event_study,
                    figures_dir,
                    [int(x) for x in plot_cfg.get("plot_relative_years", [-2, -1, 0, 1, 2, 3, 4, 5])],
                )
        if run_core_size:
            output_prefix = str(core_size_cfg.get("output_prefix", "core_size"))
            if core_size_outputs is None and panel is not None and not panel.empty:
                core_size_outputs = run_core_size_class_analysis(
                    panel=panel,
                    results_dir=did_dir,
                    outcome_cfgs=cfg.get("outcomes", {}),
                    did_cfg=did_cfg,
                    size_cfg=core_size_cfg,
                    logger=logger,
                )
            if core_size_outputs is None:
                size_event_percent_path = did_dir / f"{output_prefix}_event_study_att_percent.csv"
                size_assignment_path = did_dir / f"{output_prefix}_class_assignment.csv"
                size_summary_path = did_dir / f"{output_prefix}_class_summary.csv"
                size_group_time_path = did_dir / f"{output_prefix}_group_time_att.csv"
                if size_event_percent_path.exists():
                    core_size_outputs = {
                        "event_study_percent": pd.read_csv(size_event_percent_path),
                        "assignment": pd.read_csv(size_assignment_path) if size_assignment_path.exists() else pd.DataFrame(),
                        "summary": pd.read_csv(size_summary_path) if size_summary_path.exists() else pd.DataFrame(),
                        "group_time": pd.read_csv(size_group_time_path) if size_group_time_path.exists() else pd.DataFrame(),
                        "panel": pd.DataFrame(),
                    }
            if core_size_outputs is not None:
                if panel is not None and not panel.empty:
                    run_area_size_response_analysis(
                        event_percent=core_size_outputs.get("event_study_percent", pd.DataFrame()),
                        class_summary=core_size_outputs.get("summary", pd.DataFrame()),
                        panel=core_size_outputs.get("panel", pd.DataFrame()),
                        group_time=core_size_outputs.get("group_time", pd.DataFrame()),
                        assignment=core_size_outputs.get("assignment", pd.DataFrame()),
                        results_dir=did_dir,
                        figures_dir=figures_dir,
                        event_study_percent=percent_event_study,
                        zone_name=str(core_size_cfg.get("size_response_zone_name", "core")),
                        value_column=did_cfg.get("value_column", "mean"),
                        reference_relative_year=int(did_cfg.get("reference_relative_year", -1)),
                        figure_modes=plot_cfg.get("area_size_figures", []),
                    )
        try:
            build_fig07_fig08_outputs(
                did_dir,
                figures_dir,
                events_path=events_path,
                slope_raster_path=slope_raster_path,
                slope_all_touched=bool(recovery_cfg.get("slope_all_touched", False)),
                mapping_threshold=float(landslide_cfg.get("mask_threshold", 0.8)),
            )
        except Exception as exc:
            logger.warning("Fig.7/Fig.8 recovery tables and figures failed: %s", exc)

    logger.info("Pipeline finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run landslide ecosystem-service DID analysis.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_example.json")),
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help="Steps to run: all, events, zones, panel, did, plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation instead of using cached intermediate files.",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.steps, force=args.force)


if __name__ == "__main__":
    main()
