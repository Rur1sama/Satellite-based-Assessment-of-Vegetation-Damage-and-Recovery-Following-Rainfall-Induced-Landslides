from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from .io_utils import driver_from_path, ensure_parent


def _valid_geom(geom: Any) -> Any:
    """检查并修复几何，确保后续 buffer/difference 运算稳定。"""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom


def _ring_name(dmin: float, dmax: float) -> str:
    """根据缓冲区内外距离生成统一的 zone_name。"""
    return f"ring_{int(dmin)}_{int(dmax)}m"


def build_landslide_zones(
    events_path: str | Path,
    output_path: str | Path,
    spillover_rings_m: list[list[float]],
    study_area_path: str | Path | None = None,
    make_core_inner_zone: bool = False,
    core_inner_erosion_m: float = 5.0,
    core_inner_zone_name: str = "core_inner",
    make_never_control_zones: bool = True,
    never_control_ring_m: list[float] | None = None,
    exclude_all_landslides: bool = True,
    min_zone_area_m2: float = 0.0,
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    """
    为每个滑坡事件生成核心区、外部缓冲圈和远距离稳定控制圈。

    :param events_path: 滑坡事件矢量文件。
    :param output_path: 分析区矢量输出路径。
    :param spillover_rings_m: 破坏辐射缓冲圈，例如 [[0, 20], [20, 50]]。
    :param study_area_path: 研究区边界，所有 zone 都会裁剪到该范围。
    :param make_never_control_zones: 是否生成稳定控制圈。
    :param never_control_ring_m: 稳定控制圈距离，例如 [100, 300] 表示 100m 外。
    :param exclude_all_landslides: 是否从缓冲圈中扣除所有滑坡面。
    :param min_zone_area_m2: 小于该面积的 zone 会被丢弃。
    :param logger: 日志对象。
    :return: 分析区 GeoDataFrame。
    """
    events = gpd.read_file(events_path)
    if events.empty:
        raise ValueError(f"No events in {events_path}")
    if events.crs is None:
        raise ValueError("Events file has no CRS.")

    study_area_geom = None
    if study_area_path is not None:
        study_area = gpd.read_file(study_area_path).to_crs(events.crs)
        study_area_geom = study_area.geometry.unary_union
        if study_area_geom is None or study_area_geom.is_empty:
            raise ValueError(f"Study area is empty: {study_area_path}")

    event_sindex = events.sindex if exclude_all_landslides else None
    records: list[dict[str, Any]] = []

    for _, event in events.iterrows():
        event_geom = _valid_geom(event.geometry)
        if event_geom is None:
            continue
        event_id = str(event["event_id"])
        event_year = int(event["event_year"])
        period = str(event.get("period", event_year))

        zone_specs: list[tuple[str, float, float, Any, str, int | None]] = [
            ("core", 0.0, 0.0, event_geom, "treated", event_year)
        ]

        # 可选的坡体内部保守区：向内侵蚀若干米，剔除坡体边缘混合像元。
        # 对 10 m Sentinel-2 像元，5 m 通常相当于只保留中心落在坡体内部的像元。
        if make_core_inner_zone:
            erosion_m = float(core_inner_erosion_m)
            if erosion_m <= 0:
                raise ValueError("core_inner_erosion_m must be positive when make_core_inner_zone is true.")
            inner_core = _valid_geom(event_geom.buffer(-erosion_m))
            if inner_core is not None:
                zone_specs.append((str(core_inner_zone_name), -erosion_m, 0.0, inner_core, "treated", event_year))

        for dmin, dmax in spillover_rings_m:
            dmin = float(dmin)
            dmax = float(dmax)
            if dmax <= dmin:
                raise ValueError(f"Invalid ring distance: {dmin}-{dmax}")
            outer = event_geom.buffer(dmax)
            inner = event_geom if dmin == 0 else event_geom.buffer(dmin)
            ring = outer.difference(inner)
            if exclude_all_landslides:
                ring = _remove_intersecting_landslides(ring, events, event_sindex)
            zone_specs.append((_ring_name(dmin, dmax), dmin, dmax, ring, "treated", event_year))

        if make_never_control_zones:
            if never_control_ring_m is None or len(never_control_ring_m) != 2:
                raise ValueError("never_control_ring_m must be [inner_m, outer_m]")
            cmin = float(never_control_ring_m[0])
            cmax = float(never_control_ring_m[1])
            if cmax <= cmin:
                raise ValueError("Invalid never-control ring distances.")
            control = event_geom.buffer(cmax).difference(event_geom.buffer(cmin))
            if exclude_all_landslides:
                control = _remove_intersecting_landslides(control, events, event_sindex)
            zone_specs.append(
                (
                    f"stable_control_{int(cmin)}_{int(cmax)}m",
                    cmin,
                    cmax,
                    control,
                    "never_control",
                    None,
                )
            )

        for zone_name, dmin, dmax, geom, role, zone_event_year in zone_specs:
            if study_area_geom is not None:
                geom = geom.intersection(study_area_geom)
            geom = _valid_geom(geom)
            if geom is None:
                continue
            area = float(geom.area)
            if area < min_zone_area_m2:
                continue
            zone_id = f"{event_id}__{zone_name}"
            records.append(
                {
                    "zone_id": zone_id,
                    "event_id": event_id,
                    "matched_event_year": event_year,
                    "event_year": zone_event_year,
                    "period": period,
                    "zone_name": zone_name,
                    "zone_role": role,
                    "distance_min_m": dmin,
                    "distance_max_m": dmax,
                    "area_m2": area,
                    "geometry": geom,
                }
            )

    if not records:
        raise ValueError("No analysis zones were created.")

    zones = gpd.GeoDataFrame(records, geometry="geometry", crs=events.crs)
    zones["area_m2"] = zones.geometry.area
    zones["event_year"] = pd.to_numeric(zones["event_year"], errors="coerce")
    zones["matched_event_year"] = zones["matched_event_year"].astype(int)

    output_path = ensure_parent(output_path)
    driver = driver_from_path(output_path)
    if logger is not None:
        logger.info("Writing %s analysis zones to %s", len(zones), output_path)
    zones.to_file(output_path, driver=driver)
    return zones


def _remove_intersecting_landslides(geom: Any, events: gpd.GeoDataFrame, event_sindex: Any) -> Any:
    """
    从缓冲圈中扣除与其相交的滑坡面。

    :param geom: 待清理的缓冲圈几何。
    :param events: 全部滑坡事件。
    :param event_sindex: 滑坡事件空间索引，用于加速局部查询。
    :return: 扣除滑坡面后的几何。
    """
    geom = _valid_geom(geom)
    if geom is None or event_sindex is None:
        return geom
    candidate_idx = list(event_sindex.intersection(geom.bounds))
    if not candidate_idx:
        return geom
    candidates = events.iloc[candidate_idx]
    candidates = candidates[candidates.geometry.intersects(geom)]
    if candidates.empty:
        return geom
    return geom.difference(unary_union(candidates.geometry))
