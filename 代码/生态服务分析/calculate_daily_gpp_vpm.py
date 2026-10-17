# -*- coding: utf-8 -*-
"""
基于 VPM / 光能利用率模型计算滑坡发生当天 GPP。

输入路径
--------
1. GLASS PAR 原始数据:
   Input/EcologicalAnalysis/ExampleData/Climate_GLASS_PAR/

2. ERA5-Land Tmin / VPD / PAR / epsilon 数据:
   Input/EcologicalAnalysis/ExampleData/Climate_ERA5Land/

3. Sentinel-2 反演得到的单波段 FAPAR:
   Output/EcologicalProducts/FAPAR/

整理后的本地输入缓存
--------------------
Input/EcologicalAnalysis/WorkingCache/GPP_DailyInputs/

输出路径
--------
Output/EcologicalProducts/GPP/

主要公式
--------
GPP = PAR * FAPAR * epsilon
epsilon = epsilon_max * Tmin_scalar * VPD_scalar

单位约定
--------
PAR: MJ m-2 day-1
epsilon: kgC MJ-1
GPP: kgC m-2 day-1

说明
----
GLASS PAR 是 HDF4/HDF-EOS 文件。脚本使用 pyhdf 直接读取其中的 PAR 数据集，
不需要预先转换成 GeoTIFF。由于 pyhdf 在 Windows 下不能可靠打开含中文字符的路径，
读取时使用自动删除的 ASCII 临时副本，不在项目目录中创建持久缓存。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

try:
    from pyhdf.SD import SD, SDC
except ImportError:  # pragma: no cover - only used for a clear runtime error.
    SD = None
    SDC = None


# ==================== 路径配置 ====================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECO_INPUT_DIR = PROJECT_ROOT / "Input" / "EcologicalAnalysis"
GLASS_PAR_DIR = ECO_INPUT_DIR / "ExampleData" / "Climate_GLASS_PAR"
ERA5_LAND_DIR = ECO_INPUT_DIR / "ExampleData" / "Climate_ERA5Land"
FAPAR_DIR = PROJECT_ROOT / "Output" / "EcologicalProducts" / "FAPAR"

INPUT_ARCHIVE_DIR = ECO_INPUT_DIR / "WorkingCache" / "GPP_DailyInputs"
OUTPUT_GPP_DIR = PROJECT_ROOT / "Output" / "EcologicalProducts" / "GPP"


# ==================== 模型默认参数 ====================

# 候选 epsilon_max。单位是 kgC MJ-1 APAR。这里先给亚热带森林常见量级的候选值，
# 后续可以通过命令行 --epsilon-max 覆盖。
DEFAULT_EPSILON_MAX_VALUES = [0.0010, 0.0012, 0.001268, 0.0015]

# 当 ERA5-Land 没有现成 Tmin_scalar / VPD_scalar 时，用这些 MOD17 风格阈值计算。
DEFAULT_TMIN_MIN_C = -8.0
DEFAULT_TMIN_MAX_C = 8.0
DEFAULT_VPD_MIN_KPA = 0.65
DEFAULT_VPD_MAX_KPA = 4.0

# 输出 nodata。
OUTPUT_NODATA = -9999.0

# GLASS04B01 V4.2 is a global 0.05-degree daily mean PAR product. Its HDF
# metadata reports W m-2, so daily energy is W m-2 * 0.0864 = MJ m-2 day-1.
W_M2_TO_MJ_M2_DAY = 0.0864


DATE_RE = re.compile(r"(20\d{6})")
FAPAR_RE = re.compile(r"FAPAR_(20\d{6}).*\.tif$", re.IGNORECASE)
ERA_RE = re.compile(r"ERA5_LAND_.*?(20\d{6}).*\.tif$", re.IGNORECASE)
GLASS_RE = re.compile(r"GLASS04B01\.V\d+\.A(20\d{2})(\d{3})\..*?\.hdf$", re.IGNORECASE)


@dataclass
class DateInputs:
    """一个日期对应的全部输入文件。"""

    date: str
    fapar_path: Path
    era_path: Path
    glass_hdf_path: Path | None
    glass_xml_path: Path | None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Calculate daily GPP from PAR, FAPAR and epsilon.")
    parser.add_argument(
        "--dates",
        nargs="*",
        default=["all"],
        help="要处理的日期，格式 yyyymmdd；默认 all 表示处理 FAPAR 和 ERA5-Land 的交集。",
    )
    parser.add_argument(
        "--products",
        choices=["existing", "candidates", "both"],
        default="both",
        help="existing 使用 ERA 中已有 epsilon_kgC_MJ；candidates 用候选 epsilon_max 重新计算；both 两者都输出。",
    )
    parser.add_argument(
        "--epsilon-max",
        nargs="*",
        type=float,
        default=DEFAULT_EPSILON_MAX_VALUES,
        help="候选 epsilon_max，单位 kgC MJ-1 APAR。",
    )
    parser.add_argument(
        "--archive-mode",
        choices=["hardlink", "copy", "none"],
        default="none",
        help="是否额外归档输入文件；默认 none，不创建重复文件。",
    )
    parser.add_argument(
        "--par-scale",
        type=float,
        default=1.0,
        help="PAR 乘法校正系数。若发现 PAR_MJ_m2_day 偏高，可用 0.1 等比例重算。",
    )
    parser.add_argument(
        "--par-source",
        choices=["auto", "glass_par", "era_par", "dsr_045"],
        default="auto",
        help="PAR 来源。auto 要求直接读取 GLASS PAR；ERA5-Land 仅提供 Tmin、VPD 和 epsilon。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查匹配和元数据，不写 GPP 栅格。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 GPP 输出。",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def date_to_glass_yyyyddd(date_text: str) -> tuple[str, str]:
    """把 yyyymmdd 转成 GLASS 文件名里的 yyyy + ddd 儒略日。"""
    dt = datetime.strptime(date_text, "%Y%m%d")
    return dt.strftime("%Y"), dt.strftime("%j")


def index_fapar_files() -> dict[str, Path]:
    """索引单波段 FAPAR 文件。"""
    index: dict[str, Path] = {}
    for path in FAPAR_DIR.rglob("*.tif"):
        match = FAPAR_RE.search(path.name)
        if match:
            index[match.group(1)] = path
    return dict(sorted(index.items()))


def index_era_files() -> dict[str, Path]:
    """索引 ERA5-Land 多波段文件。"""
    index: dict[str, Path] = {}
    for path in ERA5_LAND_DIR.rglob("*.tif"):
        match = ERA_RE.search(path.name)
        if match:
            index[match.group(1)] = path
    return dict(sorted(index.items()))


def find_glass_files(date_text: str) -> tuple[Path | None, Path | None]:
    """按日期匹配 GLASS PAR HDF 和 XML。"""
    year, doy = date_to_glass_yyyyddd(date_text)
    hdf_dir = GLASS_PAR_DIR / year / "hdf"
    xml_dir = GLASS_PAR_DIR / year / "xml"
    hdf_matches = sorted(hdf_dir.glob(f"GLASS04B01*.A{year}{doy}.*.hdf")) if hdf_dir.exists() else []
    xml_matches = sorted(xml_dir.glob(f"GLASS04B01*.A{year}{doy}.*.hdf.xml")) if xml_dir.exists() else []
    return (hdf_matches[0] if hdf_matches else None, xml_matches[0] if xml_matches else None)


def build_date_inputs(requested_dates: Iterable[str]) -> list[DateInputs]:
    """根据日期建立 FAPAR / ERA / GLASS 输入清单。"""
    fapar_index = index_fapar_files()
    era_index = index_era_files()
    if list(requested_dates) == ["all"]:
        dates = sorted(set(fapar_index) & set(era_index))
    else:
        dates = sorted(set(requested_dates))

    inputs: list[DateInputs] = []
    missing_rows: list[str] = []
    for date_text in dates:
        fapar_path = fapar_index.get(date_text)
        era_path = era_index.get(date_text)
        glass_hdf, glass_xml = find_glass_files(date_text)
        if fapar_path is None or era_path is None:
            missing_rows.append(
                f"{date_text}: FAPAR={'OK' if fapar_path else 'MISSING'}, ERA={'OK' if era_path else 'MISSING'}"
            )
            continue
        inputs.append(
            DateInputs(
                date=date_text,
                fapar_path=fapar_path,
                era_path=era_path,
                glass_hdf_path=glass_hdf,
                glass_xml_path=glass_xml,
            )
        )

    if missing_rows:
        print("[WARN] Some requested dates are incomplete:")
        for row in missing_rows:
            print("  " + row)
    return inputs


def hardlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    """把输入文件整理到 Input 目录；默认优先硬链接，失败再复制。"""
    if mode == "none" or src is None:
        return
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def archive_inputs(item: DateInputs, mode: str) -> None:
    """把当天匹配到的输入文件整理到 Input/EcologicalAnalysis/WorkingCache/GPP_DailyInputs/date/。"""
    if mode == "none":
        return
    date_dir = INPUT_ARCHIVE_DIR / item.date
    hardlink_or_copy(item.fapar_path, date_dir / "FAPAR" / item.fapar_path.name, mode)
    hardlink_or_copy(item.era_path, date_dir / "ERA5_Land" / item.era_path.name, mode)
    if item.glass_hdf_path:
        hardlink_or_copy(item.glass_hdf_path, date_dir / "GLASS_PAR" / item.glass_hdf_path.name, mode)
    if item.glass_xml_path:
        hardlink_or_copy(item.glass_xml_path, date_dir / "GLASS_PAR" / item.glass_xml_path.name, mode)


def _path_is_ascii(path: Path) -> bool:
    """检查路径能否安全传给 Windows 下的 pyhdf/HDF4 C 库。"""
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


@contextmanager
def glass_hdf_read_path(item: DateInputs):
    """提供 pyhdf 可读取的 ASCII 路径；临时副本在读取后自动删除。"""
    if item.glass_hdf_path is None:
        raise FileNotFoundError(f"No GLASS PAR HDF matched date {item.date}.")
    if _path_is_ascii(item.glass_hdf_path):
        yield item.glass_hdf_path
        return

    # pyhdf 0.11.x on Windows reports "Bad file name" for paths containing
    # Chinese characters. This original-format temporary copy is deleted when
    # the context exits; it is not a converted product or persistent cache.
    with tempfile.TemporaryDirectory(prefix=f"glass_hdf_{item.date}_") as temp_dir:
        temp_path = Path(temp_dir) / item.glass_hdf_path.name
        shutil.copy2(item.glass_hdf_path, temp_path)
        yield temp_path


@contextmanager
def open_glass_par_raster(
    item: DateInputs,
    target_ds: rasterio.io.DatasetReader,
):
    """直接读取 GLASS HDF4 PAR，并以内存栅格形式提供江湾区域子集。"""
    if SD is None or SDC is None:
        raise ImportError(
            "GLASS PAR requires pyhdf. Install it in the active environment with: "
            "python -m pip install pyhdf"
        )

    path_stack = ExitStack()
    read_path = path_stack.enter_context(glass_hdf_read_path(item))
    try:
        hdf = SD(str(read_path), SDC.READ)
    except Exception:
        path_stack.close()
        raise
    try:
        datasets = hdf.datasets()
        if "PAR" not in datasets:
            raise KeyError(f"GLASS HDF has no PAR dataset. Available: {sorted(datasets)}")
        sds = hdf.select("PAR")
        attrs = sds.attributes()
        shape = tuple(int(v) for v in sds.info()[2])
        if len(shape) != 2:
            raise ValueError(f"Unexpected GLASS PAR shape: {shape}")
        height, width = shape

        xres = 360.0 / float(width)
        yres = 180.0 / float(height)
        west, south, east, north = transform_bounds(
            target_ds.crs,
            "EPSG:4326",
            *target_ds.bounds,
            densify_pts=21,
        )
        # One source-pixel margin is sufficient for bilinear resampling.
        col0 = max(0, int(math.floor((west + 180.0) / xres)) - 1)
        col1 = min(width, int(math.ceil((east + 180.0) / xres)) + 1)
        row0 = max(0, int(math.floor((90.0 - north) / yres)) - 1)
        row1 = min(height, int(math.ceil((90.0 - south) / yres)) + 1)
        if row1 <= row0 or col1 <= col0:
            raise ValueError(
                f"Target raster does not overlap GLASS global grid: {(west, south, east, north)}"
            )

        raw = sds[row0:row1, col0:col1].astype("float32")
        fill_value = float(attrs.get("_FillValue", -1))
        valid_range = attrs.get("valid_range", [0, 100000])
        valid_min, valid_max = float(valid_range[0]), float(valid_range[1])
        scale_factor = float(attrs.get("scale_factor", 1.0))
        add_offset = float(attrs.get("add_offset", 0.0))
        units = str(attrs.get("units", "")).strip()

        valid = (
            np.isfinite(raw)
            & (raw != fill_value)
            & (raw >= valid_min)
            & (raw <= valid_max)
        )
        par = np.full(raw.shape, OUTPUT_NODATA, dtype="float32")
        physical = raw[valid] * scale_factor + add_offset
        normalized_units = units.lower().replace(" ", "")
        if normalized_units in {"w/m2", "wm-2", "w/m^2"}:
            physical = physical * W_M2_TO_MJ_M2_DAY
            output_units = "MJ m-2 day-1"
            unit_conversion = "W m-2 * 0.0864"
        elif normalized_units in {
            "mj/m2/day",
            "mjm-2day-1",
            "mj/m^2/day",
        }:
            output_units = "MJ m-2 day-1"
            unit_conversion = "none"
        else:
            raise ValueError(f"Unsupported GLASS PAR units: {units!r}")
        par[valid] = physical.astype("float32")
    finally:
        hdf.end()
        path_stack.close()

    transform = from_origin(
        -180.0 + col0 * xres,
        90.0 - row0 * yres,
        xres,
        yres,
    )
    metadata = {
        "read_path": "temporary_ascii_copy" if not _path_is_ascii(item.glass_hdf_path) else str(item.glass_hdf_path),
        "dataset": "PAR",
        "source_units": units,
        "output_units": output_units,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "fill_value": fill_value,
        "valid_range": f"{valid_min},{valid_max}",
        "unit_conversion": unit_conversion,
        "subset_window": f"rows={row0}:{row1},cols={col0}:{col1}",
    }
    profile = {
        "driver": "GTiff",
        "height": par.shape[0],
        "width": par.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": OUTPUT_NODATA,
    }
    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as memory_writer:
            memory_writer.write(par, 1)
        with memory_file.open() as memory_reader:
            yield memory_reader, metadata


def band_name_map(dataset: rasterio.io.DatasetReader) -> dict[str, int]:
    """根据波段描述建立小写名称到波段号的映射。"""
    mapping: dict[str, int] = {}
    for idx, desc in enumerate(dataset.descriptions, start=1):
        if desc:
            mapping[str(desc).strip().lower()] = idx
    return mapping


def require_band(mapping: dict[str, int], candidates: list[str]) -> int:
    """按候选名称找到目标波段。"""
    for name in candidates:
        key = name.lower()
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Cannot find any band from candidates: {candidates}. Available: {sorted(mapping)}")


def calc_tmin_scalar(tmin_c: np.ndarray, tmin_min: float, tmin_max: float) -> np.ndarray:
    """根据 Tmin 计算 0-1 胁迫标量。"""
    scalar = (tmin_c - tmin_min) / (tmin_max - tmin_min)
    return np.clip(scalar, 0.0, 1.0).astype("float32")


def calc_vpd_scalar(vpd_kpa: np.ndarray, vpd_min: float, vpd_max: float) -> np.ndarray:
    """根据 VPD 计算 0-1 胁迫标量。"""
    scalar = (vpd_max - vpd_kpa) / (vpd_max - vpd_min)
    return np.clip(scalar, 0.0, 1.0).astype("float32")


def output_profile_from_fapar(fapar_ds: rasterio.io.DatasetReader) -> dict:
    """用 FAPAR 的空间网格作为 GPP 输出网格。"""
    profile = fapar_ds.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=OUTPUT_NODATA,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    )
    return profile


def safe_stats_update(values: np.ndarray, stats: dict[str, float]) -> None:
    """更新输出栅格的简要统计量。"""
    valid = np.isfinite(values) & (values != OUTPUT_NODATA)
    if not valid.any():
        return
    data = values[valid].astype("float64")
    stats["count"] += float(data.size)
    stats["sum"] += float(data.sum())
    stats["min"] = min(stats["min"], float(data.min()))
    stats["max"] = max(stats["max"], float(data.max()))


def finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    """计算统计量均值。"""
    out = stats.copy()
    out["mean"] = out["sum"] / out["count"] if out["count"] else math.nan
    return out


def write_gpp_product(
    item: DateInputs,
    product_name: str,
    epsilon_mode: str,
    epsilon_max: float | None,
    overwrite: bool,
    par_scale: float = 1.0,
    par_source: str = "auto",
    tmin_min: float = DEFAULT_TMIN_MIN_C,
    tmin_max: float = DEFAULT_TMIN_MAX_C,
    vpd_min: float = DEFAULT_VPD_MIN_KPA,
    vpd_max: float = DEFAULT_VPD_MAX_KPA,
) -> dict[str, object]:
    """逐块计算并写出一个 GPP 产品。"""
    ensure_dir(OUTPUT_GPP_DIR)
    output_path = OUTPUT_GPP_DIR / f"GPP_{item.date}_{product_name}_kgC_m2_day.tif"
    if output_path.exists() and not overwrite:
        return {
            "date": item.date,
            "product": product_name,
            "status": "exists",
            "output": str(output_path),
        }

    with rasterio.open(item.fapar_path) as fapar_ds, rasterio.open(item.era_path) as era_ds:
        era_bands = band_name_map(era_ds)
        effective_par_source = par_source
        if par_source == "auto":
            effective_par_source = "glass_par"

        par_band = None
        dsr_band = None
        if effective_par_source == "era_par":
            par_band = require_band(era_bands, ["PAR_MJ_m2_day", "par_mj_m2_day"])
        elif effective_par_source == "dsr_045":
            dsr_band = require_band(era_bands, ["DSR_MJ_m2_day", "dsr_mj_m2_day"])
        elif effective_par_source == "glass_par" and item.glass_hdf_path is None:
            raise FileNotFoundError(f"GLASS PAR was requested but no HDF matched {item.date}.")

        epsilon_band = era_bands.get("epsilon_kgc_mj")
        tmin_scalar_band = era_bands.get("tmin_scalar")
        vpd_scalar_band = era_bands.get("vpd_scalar")
        tmin_band = era_bands.get("tmin_c")
        vpd_band = era_bands.get("vpd_day_kpa")

        if epsilon_mode == "existing" and epsilon_band is None:
            raise KeyError(f"{item.era_path.name} does not contain epsilon_kgC_MJ.")
        if epsilon_mode == "candidate" and (tmin_scalar_band is None or vpd_scalar_band is None):
            if tmin_band is None or vpd_band is None:
                raise KeyError(
                    f"{item.era_path.name} lacks scalar bands and raw Tmin/VPD bands; cannot compute candidate epsilon."
                )

        profile = output_profile_from_fapar(fapar_ds)
        stats = {"count": 0.0, "sum": 0.0, "min": math.inf, "max": -math.inf}

        # WarpedVRT 按需把 ERA/GLASS 重采样到 FAPAR 的 10 m UTM 网格。
        with ExitStack() as stack:
            era_vrt = stack.enter_context(
                WarpedVRT(
                    era_ds,
                    crs=fapar_ds.crs,
                    transform=fapar_ds.transform,
                    height=fapar_ds.height,
                    width=fapar_ds.width,
                    resampling=Resampling.bilinear,
                )
            )
            glass_vrt = None
            glass_metadata: dict[str, object] = {}
            if effective_par_source == "glass_par":
                glass_ds, glass_metadata = stack.enter_context(open_glass_par_raster(item, fapar_ds))
                glass_vrt = stack.enter_context(
                    WarpedVRT(
                        glass_ds,
                        crs=fapar_ds.crs,
                        transform=fapar_ds.transform,
                        height=fapar_ds.height,
                        width=fapar_ds.width,
                        src_nodata=OUTPUT_NODATA,
                        nodata=OUTPUT_NODATA,
                        resampling=Resampling.bilinear,
                    )
                )
            out_ds = stack.enter_context(rasterio.open(output_path, "w", **profile))
            out_ds.set_band_description(1, f"GPP_{product_name}_kgC_m2_day")
            if effective_par_source == "glass_par":
                formula = "GPP = GLASS_PAR_MJ_m2_day * FAPAR * epsilon_kgC_MJ"
            elif effective_par_source == "dsr_045":
                formula = "GPP = (DSR_MJ_m2_day * 0.45) * FAPAR * epsilon_kgC_MJ"
            else:
                formula = "GPP = PAR_MJ_m2_day * FAPAR * epsilon_kgC_MJ"
            out_ds.update_tags(
                formula=formula,
                unit="kgC m-2 day-1",
                statistics_scope="full_fapar_raster",
                par_source=effective_par_source,
                par_scale=str(par_scale),
                fapar_source=str(item.fapar_path),
                era_source=str(item.era_path),
                glass_hdf_source=str(item.glass_hdf_path) if item.glass_hdf_path else "",
                glass_hdf_read_path=str(glass_metadata.get("read_path", "")),
                glass_dataset=str(glass_metadata.get("dataset", "")),
                glass_source_units=str(glass_metadata.get("source_units", "")),
                glass_scale_factor=str(glass_metadata.get("scale_factor", "")),
                glass_add_offset=str(glass_metadata.get("add_offset", "")),
                glass_unit_conversion=str(glass_metadata.get("unit_conversion", "")),
                glass_subset_window=str(glass_metadata.get("subset_window", "")),
                epsilon_mode=epsilon_mode,
                epsilon_max_kgC_MJ="" if epsilon_max is None else str(epsilon_max),
            )

            for _, window in fapar_ds.block_windows(1):
                fapar = fapar_ds.read(1, window=window).astype("float32")
                fapar = np.where((fapar >= 0.0) & (fapar <= 1.0), fapar, np.nan).astype("float32")
                if effective_par_source == "glass_par":
                    par = glass_vrt.read(1, window=window).astype("float32") * float(par_scale)
                    par = np.where(
                        np.isfinite(par) & (par >= 0.0) & (par != OUTPUT_NODATA),
                        par,
                        np.nan,
                    ).astype("float32")
                elif effective_par_source == "dsr_045":
                    par = era_vrt.read(dsr_band, window=window).astype("float32") * 0.45 * float(par_scale)
                else:
                    par = era_vrt.read(par_band, window=window).astype("float32") * float(par_scale)

                if epsilon_mode == "existing":
                    epsilon = era_vrt.read(epsilon_band, window=window).astype("float32")
                else:
                    if tmin_scalar_band is not None and vpd_scalar_band is not None:
                        tmin_scalar = era_vrt.read(tmin_scalar_band, window=window).astype("float32")
                        vpd_scalar = era_vrt.read(vpd_scalar_band, window=window).astype("float32")
                    else:
                        tmin = era_vrt.read(tmin_band, window=window).astype("float32")
                        vpd = era_vrt.read(vpd_band, window=window).astype("float32")
                        tmin_scalar = calc_tmin_scalar(tmin, tmin_min=tmin_min, tmin_max=tmin_max)
                        vpd_scalar = calc_vpd_scalar(vpd, vpd_min=vpd_min, vpd_max=vpd_max)
                    epsilon = float(epsilon_max) * tmin_scalar * vpd_scalar

                gpp = par * fapar * epsilon
                gpp = np.where(np.isfinite(gpp), gpp, OUTPUT_NODATA).astype("float32")
                safe_stats_update(gpp, stats)
                out_ds.write(gpp, 1, window=window)

    final = finalize_stats(stats)
    return {
        "date": item.date,
        "product": product_name,
        "status": "written",
        "output": str(output_path),
        "count": int(final["count"]),
        "min": final["min"],
        "max": final["max"],
        "mean": final["mean"],
        "epsilon_mode": epsilon_mode,
        "epsilon_max": epsilon_max,
        "par_source": effective_par_source,
        "statistics_scope": "full_fapar_raster",
    }


def inspect_inputs(items: list[DateInputs]) -> None:
    """打印输入匹配和关键元数据。"""
    print("[INFO] Matched dates:")
    for item in items:
        print(f"  {item.date}")
        print(f"    FAPAR: {item.fapar_path}")
        print(f"    ERA:   {item.era_path}")
        print(f"    GLASS: {item.glass_hdf_path if item.glass_hdf_path else 'MISSING/UNREAD'}")

    if items:
        with rasterio.open(items[0].fapar_path) as fapar_ds:
            print("[INFO] FAPAR sample metadata:")
            print(f"  crs={fapar_ds.crs}, res={fapar_ds.res}, shape=({fapar_ds.height}, {fapar_ds.width})")
            print(f"  descriptions={fapar_ds.descriptions}, tags={fapar_ds.tags(1)}")
        with rasterio.open(items[0].era_path) as era_ds:
            print("[INFO] ERA sample metadata:")
            print(f"  crs={era_ds.crs}, res={era_ds.res}, shape=({era_ds.height}, {era_ds.width})")
            print(f"  descriptions={era_ds.descriptions}")


def write_summary(rows: list[dict[str, object]]) -> None:
    """写出 GPP 产品清单和统计摘要。"""
    ensure_dir(OUTPUT_GPP_DIR)
    summary_path = OUTPUT_GPP_DIR / "daily_gpp_summary.csv"
    keys = sorted({key for row in rows for key in row.keys()})
    candidates = [
        summary_path,
        OUTPUT_GPP_DIR / "daily_gpp_summary_latest.csv",
        OUTPUT_GPP_DIR / f"daily_gpp_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            with candidate.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO] Summary written: {candidate}")
            return
        except PermissionError as exc:
            last_error = exc
    raise PermissionError(f"Cannot write summary CSV. Last error: {last_error}")


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    items = build_date_inputs(args.dates)
    inspect_inputs(items)
    if args.dry_run:
        print("[INFO] Dry run finished. No GPP raster was written.")
        return

    rows: list[dict[str, object]] = []
    for item in items:
        print(f"[INFO] Processing {item.date}")
        archive_inputs(item, args.archive_mode)
        if args.products in ["existing", "both"]:
            rows.append(
                write_gpp_product(
                    item=item,
                    product_name="existing_epsilon",
                    epsilon_mode="existing",
                    epsilon_max=None,
                    overwrite=args.overwrite,
                    par_scale=args.par_scale,
                    par_source=args.par_source,
                )
            )
        if args.products in ["candidates", "both"]:
            for eps in args.epsilon_max:
                eps_label = str(eps).replace(".", "p")
                rows.append(
                    write_gpp_product(
                        item=item,
                        product_name=f"epsmax_{eps_label}",
                        epsilon_mode="candidate",
                        epsilon_max=float(eps),
                        overwrite=args.overwrite,
                        par_scale=args.par_scale,
                        par_source=args.par_source,
                    )
                )
    write_summary(rows)


if __name__ == "__main__":
    main()
