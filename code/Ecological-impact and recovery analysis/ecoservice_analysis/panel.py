from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import mapping

from .io_utils import ensure_parent


PANEL_COLUMNS = [
    "outcome",
    "year",
    "unit_id",
    "zone_id",
    "event_id",
    "zone_name",
    "zone_role",
    "event_year",
    "matched_event_year",
    "relative_year",
    "distance_min_m",
    "distance_max_m",
    "area_m2",
    "n_pixels",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "raster_path",
]

CACHE_VERSION = 3


def _safe_cache_token(value: str) -> str:
    """把 outcome 等名称转成可用于 Windows 文件名的短 token。"""
    token = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value)).strip("_")
    return token or "unknown"


def _json_default(value: Any) -> Any:
    """JSON 写缓存元数据时兼容 Path 和 numpy 标量。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _frame_fingerprint(df: pd.DataFrame) -> str:
    """对当前 zone 元数据做稳定指纹，事件重编号或面积变化时自动失效旧缓存。"""
    if df.empty:
        return "empty"
    text = df.sort_values("unit_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _zone_cache_meta(zones: gpd.GeoDataFrame) -> tuple[pd.DataFrame, str]:
    '''
    提取会影响掩膜统计结果的 zone 元数据和稳定指纹。
    指纹同时包含属性和几何，避免事件边界改变但编号、面积近似不变时误用旧缓存。

    :param zones: 当前参与栅格统计的分析区 GeoDataFrame。
    :return: 用于缓存校验的元数据表及其 SHA-256 指纹。
    '''
    cols = [
        col
        for col in [
            "zone_id",
            "event_id",
            "event_year",
            "matched_event_year",
            "zone_name",
            "zone_role",
            "distance_min_m",
            "distance_max_m",
            "area_m2",
        ]
        if col in zones.columns
    ]
    meta = zones[cols].copy()
    meta["unit_id"] = meta["zone_id"].astype(str)
    for col in ["zone_id", "event_id", "zone_name", "zone_role"]:
        if col in meta.columns:
            meta[col] = meta[col].astype(str)
    for col in ["event_year", "matched_event_year", "distance_min_m", "distance_max_m", "area_m2"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce").round(6)
    meta["geometry_sha256"] = zones.geometry.apply(
        lambda geom: hashlib.sha256(bytes(geom.wkb)).hexdigest()
        if geom is not None and not geom.is_empty
        else "empty"
    )
    meta["geometry_crs"] = str(zones.crs)
    return meta, _frame_fingerprint(meta)


def _raster_fingerprint(paths: list[Path]) -> list[dict[str, Any]]:
    """记录参与该年提取的栅格文件状态，栅格被替换后自动重算。"""
    items: list[dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        stat = p.stat()
        items.append(
            {
                "path": str(p),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return items


def _spec_fingerprint(
    raster_paths: list[Path],
    band: int,
    scale: float,
    offset: float,
    all_touched: bool,
    min_valid_pixels: int,
) -> str:
    """对单个 outcome-year 的提取参数做指纹。"""
    payload = {
        "cache_version": CACHE_VERSION,
        "rasters": _raster_fingerprint(raster_paths),
        "band": int(band),
        "scale": float(scale),
        "offset": float(offset),
        "all_touched": bool(all_touched),
        "min_valid_pixels": int(min_valid_pixels),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_paths(
    cache_dir: Path,
    outcome_name: str,
    year: int,
    zone_hash: str,
    spec_hash: str,
) -> tuple[Path, Path]:
    """构造 outcome-year 分片缓存路径。"""
    name = f"{_safe_cache_token(outcome_name)}_{int(year)}_{zone_hash[:12]}_{spec_hash[:10]}"
    return cache_dir / f"{name}.csv", cache_dir / f"{name}.json"


def _read_year_cache(
    cache_csv: Path,
    cache_json: Path,
    outcome_name: str,
    year: int,
    zone_hash: str,
    spec_hash: str,
    logger: logging.Logger | None,
) -> list[dict[str, Any]] | None:
    """读取已完成的 outcome-year 缓存；元数据不匹配则返回 None。"""
    if not cache_csv.exists() or not cache_json.exists():
        return None
    try:
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    expected = {
        "cache_version": CACHE_VERSION,
        "outcome": str(outcome_name),
        "year": int(year),
        "zone_hash": str(zone_hash),
        "spec_hash": str(spec_hash),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            return None
    try:
        cached = pd.read_csv(cache_csv)
    except Exception:
        return None
    cached = cached.reindex(columns=PANEL_COLUMNS)
    if logger is not None:
        logger.info("Use cached extraction for %s %s: %s rows", outcome_name, year, len(cached))
    return cached.to_dict("records")


def _write_year_cache(
    rows: list[dict[str, Any]],
    cache_csv: Path,
    cache_json: Path,
    outcome_name: str,
    year: int,
    zone_hash: str,
    spec_hash: str,
    n_zones: int,
) -> None:
    """原子写入 outcome-year 缓存，避免中断时留下半截 CSV。"""
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    cache_json.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    tmp_csv = cache_csv.with_suffix(cache_csv.suffix + ".tmp")
    tmp_json = cache_json.with_suffix(cache_json.suffix + ".tmp")
    frame.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
    meta = {
        "cache_version": CACHE_VERSION,
        "outcome": str(outcome_name),
        "year": int(year),
        "zone_hash": str(zone_hash),
        "spec_hash": str(spec_hash),
        "n_zones": int(n_zones),
        "n_rows": int(len(frame)),
    }
    tmp_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_csv.replace(cache_csv)
    tmp_json.replace(cache_json)


def _zonal_stats_for_geometry(
    src: rasterio.io.DatasetReader,
    geom: Any,
    band: int = 1,
    all_touched: bool = False,
    scale: float = 1.0,
    offset: float = 0.0,
) -> dict[str, float | int | None]:
    """
    对单个矢量区在单幅栅格上做分区统计。

    :param src: 已打开的 rasterio 栅格对象。
    :param geom: 需要统计的矢量几何，坐标系必须与 src 一致。
    :param band: 栅格波段序号。
    :param all_touched: 是否统计所有被几何边界触碰到的像元。
    :param scale: 像元值缩放系数。
    :param offset: 像元值偏移量。
    :return: 有效像元数、均值、中位数、标准差、最小值和最大值。
    """
    if geom is None or geom.is_empty:
        return _empty_stats()
    try:
        window = geometry_window(src, [mapping(geom)])
    except (WindowError, ValueError):
        return _empty_stats()
    if window.width <= 0 or window.height <= 0:
        return _empty_stats()

    arr = src.read(band, window=window, masked=True)
    transform = src.window_transform(window)
    inside = geometry_mask(
        [mapping(geom)],
        out_shape=arr.shape,
        transform=transform,
        invert=True,
        all_touched=all_touched,
    )

    mask = ~inside
    if np.ma.isMaskedArray(arr):
        mask = np.logical_or(mask, np.ma.getmaskarray(arr))
        values = np.ma.array(arr, mask=mask).compressed()
    else:
        values = arr[inside]

    values = values.astype("float64", copy=False)
    if src.nodata is not None:
        values = values[values != src.nodata]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_stats()

    values = values * float(scale) + float(offset)
    return {
        "n_pixels": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _empty_stats() -> dict[str, float | int | None]:
    """返回无有效像元时的空统计结果。"""
    return {
        "n_pixels": 0,
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
    }


def extract_zone_outcome_panel(
    zones_path: str | Path,
    outcome_specs: dict[str, dict[str, Any]],
    output_csv: str | Path,
    all_touched: bool = False,
    min_valid_pixels: int = 1,
    unit_ids: set[str] | None = None,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    提取“空间单元-年份-生态变量”的面板数据，并保存为 CSV 缓存。

    :param zones_path: 滑坡体、缓冲区和稳定控制区矢量文件。
    :param outcome_specs: 生态变量配置，包含每年一幅或多幅栅格路径。
    :param output_csv: 面板数据输出路径；下次可直接复用，不必重新按掩膜提取。
    :param all_touched: 是否采用 all_touched 栅格掩膜规则。
    :param min_valid_pixels: 每个空间单元至少需要的有效像元数。
    :param unit_ids: 仅提取指定空间单元；为 None 时提取全部空间单元。
    :param cache_dir: outcome-year 分片缓存目录。
    :param reuse_cache: 是否允许读取已有分片缓存；False 时重新提取并覆盖当前指纹缓存。
    :param logger: 日志对象。
    :return: 面板数据 DataFrame。
    """
    zones = gpd.read_file(zones_path)
    if zones.empty:
        raise ValueError(f"No zones in {zones_path}")
    if zones.crs is None:
        raise ValueError("Zones file has no CRS.")
    if unit_ids is not None:
        # 增量提取时只保留缺失的 zone_id，避免每次新增一个分析口径就重跑全部掩膜统计。
        unit_ids = {str(unit_id) for unit_id in unit_ids}
        zones = zones[zones["zone_id"].astype(str).isin(unit_ids)].copy()
        if zones.empty:
            panel = pd.DataFrame(columns=PANEL_COLUMNS)
            output_csv = ensure_parent(output_csv)
            panel.to_csv(output_csv, index=False, encoding="utf-8-sig")
            return panel

    output_csv = ensure_parent(output_csv)
    if cache_dir is None:
        cache_dir = output_csv.with_name(f"{output_csv.stem}_outcome_year_cache")
    cache_dir = Path(cache_dir)

    zone_meta, zone_hash = _zone_cache_meta(zones)
    rows: list[dict[str, Any]] = []
    for outcome_name, spec in outcome_specs.items():
        rasters: dict[int, list[Path]] = spec["rasters"]
        band = int(spec.get("band", 1))
        scale = float(spec.get("scale", 1.0))
        offset = float(spec.get("offset", 0.0))
        outcome_all_touched = bool(spec.get("all_touched", all_touched))

        for year, raster_paths in sorted(rasters.items()):
            raster_paths = [Path(path) for path in raster_paths]
            spec_hash = _spec_fingerprint(
                raster_paths=raster_paths,
                band=band,
                scale=scale,
                offset=offset,
                all_touched=outcome_all_touched,
                min_valid_pixels=min_valid_pixels,
            )
            cache_csv, cache_json = _cache_paths(
                cache_dir=cache_dir,
                outcome_name=outcome_name,
                year=int(year),
                zone_hash=zone_hash,
                spec_hash=spec_hash,
            )
            cached_rows = None
            if reuse_cache:
                cached_rows = _read_year_cache(
                    cache_csv=cache_csv,
                    cache_json=cache_json,
                    outcome_name=outcome_name,
                    year=int(year),
                    zone_hash=zone_hash,
                    spec_hash=spec_hash,
                    logger=logger,
                )
            if cached_rows is not None:
                rows.extend(cached_rows)
                continue

            tile_rows: list[dict[str, Any]] = []
            for raster_path in raster_paths:
                if logger is not None:
                    logger.info("Extracting %s for %s from %s", outcome_name, year, raster_path)
                with rasterio.open(raster_path) as src:
                    if src.crs is None:
                        raise ValueError(f"Raster has no CRS: {raster_path}")
                    zones_raster_crs = zones.to_crs(src.crs)
                    for idx, zone in zones_raster_crs.iterrows():
                        stats = _zonal_stats_for_geometry(
                            src=src,
                            geom=zone.geometry,
                            band=band,
                            all_touched=outcome_all_touched,
                            scale=scale,
                            offset=offset,
                        )
                        if int(stats["n_pixels"] or 0) < min_valid_pixels:
                            continue

                        event_year = zone.get("event_year")
                        rel_year = None
                        if pd.notna(event_year):
                            rel_year = int(year) - int(event_year)

                        original_zone = zones.loc[idx]
                        tile_rows.append(
                            {
                                "outcome": outcome_name,
                                "year": int(year),
                                "unit_id": str(original_zone["zone_id"]),
                                "zone_id": str(original_zone["zone_id"]),
                                "event_id": str(original_zone["event_id"]),
                                "zone_name": str(original_zone["zone_name"]),
                                "zone_role": str(original_zone["zone_role"]),
                                "event_year": event_year if pd.notna(event_year) else None,
                                "matched_event_year": int(original_zone["matched_event_year"]),
                                "relative_year": rel_year,
                                "distance_min_m": float(original_zone["distance_min_m"]),
                                "distance_max_m": float(original_zone["distance_max_m"]),
                                "area_m2": float(original_zone["area_m2"]),
                                "raster_path": str(raster_path),
                                **stats,
                            }
                        )
            year_rows = _aggregate_tiles(tile_rows)
            _write_year_cache(
                rows=year_rows,
                cache_csv=cache_csv,
                cache_json=cache_json,
                outcome_name=outcome_name,
                year=int(year),
                zone_hash=zone_hash,
                spec_hash=spec_hash,
                n_zones=len(zone_meta),
            )
            rows.extend(year_rows)

    panel = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    panel.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if logger is not None:
        logger.info("Wrote panel with %s rows to %s", len(panel), output_csv)
    return panel


def _aggregate_tiles(tile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    合并同一年多个 Sentinel-2 tile 的统计结果。

    :param tile_rows: 单 tile 提取得到的记录。
    :return: 按有效像元数加权后的年度记录，避免同一年多 tile 被当成重复观测。
    """
    if not tile_rows:
        return []
    df = pd.DataFrame(tile_rows)
    group_cols = [
        "outcome",
        "year",
        "unit_id",
        "zone_id",
        "event_id",
        "zone_name",
        "zone_role",
        "event_year",
        "matched_event_year",
        "relative_year",
        "distance_min_m",
        "distance_max_m",
        "area_m2",
    ]
    out_rows: list[dict[str, Any]] = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        weights = chunk["n_pixels"].astype(float)
        total_pixels = int(weights.sum())
        if total_pixels <= 0:
            continue
        values = {}
        for col in ["mean", "median", "std", "min", "max"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        values["mean"] = float(np.average(chunk["mean"], weights=weights))
        values["median"] = float(np.average(chunk["median"], weights=weights))
        values["std"] = float(np.sqrt(np.average(chunk["std"].fillna(0.0) ** 2, weights=weights)))
        values["min"] = float(chunk["min"].min())
        values["max"] = float(chunk["max"].max())
        out_rows.append(
            {
                **base,
                "n_pixels": total_pixels,
                **values,
                "raster_path": ";".join(chunk["raster_path"].astype(str).tolist()),
            }
        )
    return out_rows
