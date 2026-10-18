from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from shapely.geometry import Polygon, box, shape

from .io_utils import driver_from_path, ensure_parent


def _estimate_metric_crs_from_raster(path: Path) -> Any:
    """根据栅格范围估算一个适合距离缓冲分析的 UTM 投影坐标系。"""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        bounds_geom = box(*src.bounds)
        bounds_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bounds_geom], crs=src.crs)
        estimated = bounds_gdf.estimate_utm_crs()
        if estimated is None:
            raise ValueError("Could not estimate a metric CRS; set analysis_crs in config.")
        return estimated


def _clean_polygon(geom: Any) -> Any:
    """修复无效几何，并把空几何统一转成 None。"""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom


def choose_analysis_crs(mask_specs: list[dict[str, Any]], analysis_crs: str | None) -> Any:
    """
    确定分析坐标系。

    :param mask_specs: 滑坡掩膜配置列表。
    :param analysis_crs: 用户手动指定的坐标系；为空时自动判断。
    :return: 用于面积和距离计算的 CRS。
    """
    if analysis_crs:
        return analysis_crs
    if not mask_specs:
        raise ValueError("No landslide masks were found.")
    first_path = Path(mask_specs[0]["path"])
    with rasterio.open(first_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {first_path}")
        if not src.crs.is_geographic:
            return src.crs
    return _estimate_metric_crs_from_raster(first_path)


def masks_to_landslide_events(
    mask_specs: list[dict[str, Any]],
    output_path: str | Path,
    analysis_crs: str | None = None,
    study_area_path: str | Path | None = None,
    mask_threshold: float = 0.80,  # 概率不低于该阈值的像元判定为滑坡
    min_area_m2: float = 0.0,
    connectivity: int = 8, # 一个滑坡像素无论是上下左右相连，还是对角线相连，都会被算作同一个滑坡体
    logger: logging.Logger | None = None,
) -> gpd.GeoDataFrame:
    '''
    把年度滑坡掩膜 tif 转成滑坡事件矢量面。

    :param mask_specs: 每个年度掩膜的路径、年份对和事件年份。
    :param output_path: 事件面输出路径。
    :param analysis_crs: 分析坐标系；为空时自动估计。
    :param study_area_path: 研究区边界，事件面会裁剪到该范围内。
    :param mask_threshold: 掩膜阈值，概率不低于该值时视为滑坡。
    :param min_area_m2: 小于该面积的斑块会被过滤。
    :param connectivity: 栅格连通性，4 或 8。
    :param logger: 日志对象。
    :return: 滑坡事件 GeoDataFrame。
    '''
    if not mask_specs:
        raise ValueError("No existing landslide masks are available.")

    target_crs = choose_analysis_crs(mask_specs, analysis_crs)
    study_area_geom = None
    if study_area_path is not None:
        study_area = gpd.read_file(study_area_path).to_crs(target_crs)
        study_area_geom = study_area.geometry.unary_union
        if study_area_geom is None or study_area_geom.is_empty:
            raise ValueError(f"Study area is empty: {study_area_path}")

    all_frames: list[gpd.GeoDataFrame] = []
    global_index = 1

    for spec in mask_specs:
        path = Path(spec["path"])
        event_year = int(spec["event_year"])
        period = str(spec.get("period", event_year))
        if logger is not None:
            logger.info("Vectorizing landslide mask %s (%s)", period, path)

        records: list[dict[str, Any]] = []
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")
            arr = src.read(1, masked=True)
            valid = np.ma.filled(arr, 0) >= mask_threshold
            if not np.any(valid):
                if logger is not None:
                    logger.warning("No landslide pixels in mask: %s", path)
                continue

            for geom_mapping, value in features.shapes(
                valid.astype("uint8"),
                mask=valid,
                transform=src.transform,
                connectivity=connectivity,
            ):
                if int(value) != 1:
                    continue
                geom = _clean_polygon(shape(geom_mapping))
                if geom is None:
                    continue
                records.append(
                    {
                        "event_year": event_year,
                        "period": period,
                        "source_mask": str(path),
                        "geometry": geom,
                    }
                )

            if not records:
                continue

            gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=src.crs).to_crs(target_crs)

        gdf["geometry"] = gdf.geometry.apply(_clean_polygon)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        if study_area_geom is not None:
            gdf["geometry"] = gdf.geometry.intersection(study_area_geom).apply(_clean_polygon)
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf["area_m2"] = gdf.geometry.area
        if min_area_m2 > 0:
            gdf = gdf[gdf["area_m2"] >= float(min_area_m2)].copy()
        if gdf.empty:
            if logger is not None:
                logger.warning("No events retained after area filtering for %s", path)
            continue

        ids = []
        for _ in range(len(gdf)):
            ids.append(f"LS_{event_year}_{global_index:06d}")
            global_index += 1
        gdf.insert(0, "event_id", ids)
        all_frames.append(gdf)

    if not all_frames:
        raise ValueError("No landslide event polygons were created.")

    events = gpd.GeoDataFrame(pd.concat(all_frames, ignore_index=True), crs=target_crs)
    events["event_year"] = events["event_year"].astype(int)
    events["area_m2"] = events.geometry.area

    output_path = ensure_parent(output_path)
    driver = driver_from_path(output_path)
    if logger is not None:
        logger.info("Writing %s landslide events to %s", len(events), output_path)
    events.to_file(output_path, driver=driver)
    return events
