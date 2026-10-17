from __future__ import annotations

import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


ROOT = Path(r"E:\python_base\Slide_dect")
RAW_L2A_DIR = Path(r"E:\Raster_data\Sentinel-2\韶关市江湾镇\L2A_raw")
SNAP_RESULTS_DIR = ROOT / "Output" / "SNAP_Results"
BIOPHYSICAL_DIR = SNAP_RESULTS_DIR / "Biophysical"
SINGLE_BAND_DIR = SNAP_RESULTS_DIR / "Biophysical_SingleBand"
LAI_SINGLE_DIR = SINGLE_BAND_DIR / "LAI"
DID_DIR = ROOT / "Output" / "EcoService_DID"
AUDIT_DIR = DID_DIR / "06_snap_lai_qc_audit"

VAR_BANDS = {"LAI": 1, "FAPAR": 3, "FVC": 5}
QC_BANDS = {"LAI": 2, "FAPAR": 4, "FVC": 6}

FLAG_BITS = {
    0: "INPUT_OUT_OF_RANGE",
    1: "OUTPUT_THRESHOLDED_TO_MIN_OUTPUT",
    2: "OUTPUT_THRESHOLDED_TO_MAX_OUTPUT",
    3: "OUTPUT_TOO_LOW",
    4: "OUTPUT_TOO_HIGH",
}

SCL_CLASSES = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "CAST_SHADOWS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW_OR_ICE",
}

BIO_RE = re.compile(
    r"^Biophysical_(?P<date>\d{8})_(?P<baseline>N\d+)_(?P<orbit>R\d+)_"
    r"(?P<tile>T\d{2}[A-Z]{3})_(?P<sensor>S2[A-Z]_10m)\.tif$",
    re.IGNORECASE,
)
S2_RE = re.compile(
    r"^(?P<mission>S2[A-Z])_MSIL2A_(?P<datetime>\d{8}T\d{6})_"
    r"(?P<baseline>N\d+)_(?P<orbit>R\d+)_(?P<tile>T\d{2}[A-Z]{3})_",
    re.IGNORECASE,
)


def parse_s2_name(name: str) -> dict[str, str]:
    match = S2_RE.match(name)
    if not match:
        return {
            "mission": "UNKNOWN",
            "date": "UNKNOWN",
            "baseline": "UNKNOWN",
            "orbit": "UNKNOWN",
            "tile": "UNKNOWN",
        }
    info = match.groupdict()
    info["date"] = info["datetime"][:8]
    return info


def parse_biophysical_name(name: str) -> dict[str, str]:
    match = BIO_RE.match(name)
    if not match:
        return {
            "date": "UNKNOWN",
            "baseline": "UNKNOWN",
            "orbit": "UNKNOWN",
            "tile": "UNKNOWN",
            "sensor": "UNKNOWN",
            "mission": "UNKNOWN",
        }
    info = match.groupdict()
    info["mission"] = info["sensor"][:3]
    return info


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_inventory(root: Path, label: str, skip_parts: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip_parts = skip_parts or set()
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        stat = path.stat()
        rows.append(
            {
                "scope": label,
                "relative_path": safe_rel(path),
                "absolute_path": str(path),
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "last_write_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def find_scl_20m_in_zip(zip_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return None
    scl = [name for name in names if name.endswith("_SCL_20m.jp2") and "/IMG_DATA/R20m/" in name]
    if scl:
        return sorted(scl)[0]
    fallback = [name for name in names if name.endswith("_SCL_20m.jp2")]
    return sorted(fallback)[0] if fallback else None


def build_raw_inventory() -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not RAW_L2A_DIR.exists():
        return pd.DataFrame(), index

    for path in sorted(RAW_L2A_DIR.iterdir()):
        if not (path.name.endswith(".zip") or path.name.endswith(".SAFE")):
            continue
        meta = parse_s2_name(path.name)
        scl_inner = None
        if path.suffix.lower() == ".zip":
            scl_inner = find_scl_20m_in_zip(path)
        row = {
            **meta,
            "file_name": path.name,
            "path": str(path),
            "kind": path.suffix.lower().lstrip(".") or "SAFE",
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "scl_20m_inner_path": scl_inner,
            "has_scl_20m": bool(scl_inner),
        }
        rows.append(row)
        index[(meta["mission"], meta["date"], meta["tile"])] = row
    return pd.DataFrame(rows), index


def vsizip_path(zip_path: str | Path, inner: str) -> str:
    return "/vsizip/" + Path(zip_path).as_posix() + "/" + inner


def make_bins(variable: str) -> np.ndarray:
    if variable == "LAI":
        return np.concatenate(([-1000.0], np.linspace(0.0, 8.0, 161), [1000.0]))
    return np.concatenate(([-1000.0], np.linspace(0.0, 1.0, 101), [1000.0]))


def hist_quantile(hist: np.ndarray, bins: np.ndarray, q: float) -> float | None:
    total = int(hist.sum())
    if total == 0:
        return None
    target = q * total
    csum = np.cumsum(hist)
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = max(0, min(idx, len(bins) - 2))
    left = bins[idx]
    right = bins[idx + 1]
    if left <= -999:
        return float(right)
    if right >= 999:
        return float(left)
    return float((left + right) / 2.0)


@dataclass
class RunningStats:
    variable: str
    bins: np.ndarray = field(init=False)
    hist: np.ndarray = field(init=False)
    count: int = 0
    finite_count: int = 0
    sum_value: float = 0.0
    sum_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf
    gt_0_8: int = 0
    gt_1_0: int = 0
    gt_2_0: int = 0

    def __post_init__(self) -> None:
        self.bins = make_bins(self.variable)
        self.hist = np.zeros(len(self.bins) - 1, dtype=np.int64)

    def update(self, values: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        arr = np.asarray(values)
        if valid_mask is None:
            mask = np.isfinite(arr)
        else:
            mask = valid_mask & np.isfinite(arr)
        data = arr[mask]
        if data.size == 0:
            return
        data64 = data.astype("float64", copy=False)
        self.count += int(data64.size)
        self.finite_count += int(data64.size)
        self.sum_value += float(data64.sum())
        self.sum_sq += float(np.square(data64).sum())
        self.min_value = min(self.min_value, float(data64.min()))
        self.max_value = max(self.max_value, float(data64.max()))
        self.hist += np.histogram(data64, bins=self.bins)[0]
        if self.variable == "LAI":
            self.gt_0_8 += int((data64 > 0.8).sum())
            self.gt_1_0 += int((data64 > 1.0).sum())
            self.gt_2_0 += int((data64 > 2.0).sum())

    def as_dict(self) -> dict[str, Any]:
        if self.finite_count == 0:
            mean = std = None
            min_value = max_value = None
        else:
            mean = self.sum_value / self.finite_count
            var = max(0.0, self.sum_sq / self.finite_count - mean * mean)
            std = math.sqrt(var)
            min_value = self.min_value
            max_value = self.max_value
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "min": min_value,
            "p05": hist_quantile(self.hist, self.bins, 0.05),
            "p25": hist_quantile(self.hist, self.bins, 0.25),
            "median": hist_quantile(self.hist, self.bins, 0.50),
            "p75": hist_quantile(self.hist, self.bins, 0.75),
            "p95": hist_quantile(self.hist, self.bins, 0.95),
            "max": max_value,
            "mean": mean,
            "std": std,
            "lai_gt_0_8_count": self.gt_0_8 if self.variable == "LAI" else None,
            "lai_gt_1_0_count": self.gt_1_0 if self.variable == "LAI" else None,
            "lai_gt_2_0_count": self.gt_2_0 if self.variable == "LAI" else None,
        }


@dataclass
class FlagStats:
    total_count: int = 0
    zero_count: int = 0
    bit_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    unique_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, values: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        arr = np.asarray(values)
        mask = np.isfinite(arr)
        if valid_mask is not None:
            mask &= valid_mask
        data = arr[mask]
        if data.size == 0:
            return
        data_i = np.rint(data).astype(np.int64, copy=False)
        self.total_count += int(data_i.size)
        self.zero_count += int((data_i == 0).sum())
        unique, counts = np.unique(data_i, return_counts=True)
        for value, count in zip(unique, counts):
            self.unique_counts[int(value)] += int(count)
        for bit in FLAG_BITS:
            self.bit_counts[bit] += int(((data_i & (1 << bit)) != 0).sum())


def masked_data_and_valid(masked: np.ma.MaskedArray, nodata: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(masked)
    mask = ~np.ma.getmaskarray(masked)
    if nodata is not None and np.isfinite(nodata):
        mask &= data != nodata
    mask &= np.isfinite(data)
    return data, mask


def split_missing_lai(multiband_paths: list[Path]) -> pd.DataFrame:
    LAI_SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for tif_path in multiband_paths:
        expected = LAI_SINGLE_DIR / tif_path.name.replace("Biophysical_", "LAI_", 1)
        row = {
            "source_multiband": str(tif_path),
            "expected_lai": str(expected),
            "already_existed": expected.exists(),
            "written_now": False,
        }
        if not expected.exists():
            with rasterio.open(tif_path) as src:
                if src.count < 1:
                    row["error"] = "source has no band"
                else:
                    profile = src.profile.copy()
                    profile.update(
                        count=1,
                        driver="GTiff",
                        compress="deflate",
                        predictor=2,
                        tiled=True,
                        BIGTIFF="IF_SAFER",
                    )
                    with rasterio.open(expected, "w", **profile) as dst:
                        for _, window in src.block_windows(1):
                            dst.write(src.read(1, window=window), 1, window=window)
                        dst.set_band_description(1, "LAI")
                        dst.update_tags(
                            product="LAI",
                            source_file=tif_path.name,
                            source_band="1",
                            source_band_order="1=LAI,2=LAIQC,3=FAPAR,4=FAPARQC,5=FVC,6=FVCQC",
                        )
                    row["written_now"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def audit_biophysical_products(raw_index: dict[tuple[str, str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multiband_paths = sorted(BIOPHYSICAL_DIR.glob("Biophysical_*.tif"))
    scene_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    scl_rows: list[dict[str, Any]] = []

    for scene_i, tif_path in enumerate(multiband_paths, start=1):
        meta = parse_biophysical_name(tif_path.name)
        raw = raw_index.get((meta["mission"], meta["date"], meta["tile"]))
        matched_scl = bool(raw and raw.get("has_scl_20m"))
        print(f"[{scene_i}/{len(multiband_paths)}] auditing {tif_path.name}; matched_scl={matched_scl}")

        scene_stats = {var: RunningStats(var) for var in VAR_BANDS}
        flag_stats = {var: FlagStats() for var in VAR_BANDS}
        scl_stats: dict[tuple[str, int], RunningStats] = {}
        scl_counts: dict[int, int] = defaultdict(int)

        with rasterio.open(tif_path) as src:
            scene_row = {
                **meta,
                "file_name": tif_path.name,
                "path": str(tif_path),
                "size_bytes": tif_path.stat().st_size,
                "band_count": src.count,
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "transform": tuple(src.transform)[:6],
                "nodata": src.nodata,
                "matched_raw_l2a": bool(raw),
                "matched_scl_20m": matched_scl,
                "raw_l2a_file": raw.get("file_name") if raw else None,
            }
            scl_vrt = None
            scl_src = None
            if matched_scl and raw:
                scl_src = rasterio.open(vsizip_path(raw["path"], raw["scl_20m_inner_path"]))
                scl_vrt = WarpedVRT(
                    scl_src,
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    resampling=Resampling.nearest,
                )

            try:
                for _, window in src.block_windows(1):
                    for var, band in VAR_BANDS.items():
                        data_ma = src.read(band, window=window, masked=True)
                        data, valid = masked_data_and_valid(data_ma, src.nodata)
                        scene_stats[var].update(data, valid)

                        qc_band = QC_BANDS[var]
                        if src.count >= qc_band:
                            qc_ma = src.read(qc_band, window=window, masked=True)
                            qc, qc_valid = masked_data_and_valid(qc_ma, src.nodata)
                            flag_stats[var].update(qc, qc_valid)

                        if scl_vrt is not None:
                            scl_ma = scl_vrt.read(1, window=window, masked=True)
                            scl_data = np.ma.getdata(scl_ma)
                            scl_valid = ~np.ma.getmaskarray(scl_ma)
                            joint_valid = valid & scl_valid
                            if joint_valid.any():
                                unique_scl = np.unique(scl_data[joint_valid])
                                for scl_code_raw in unique_scl:
                                    scl_code = int(scl_code_raw)
                                    mask = joint_valid & (scl_data == scl_code)
                                    if var == "LAI":
                                        scl_counts[scl_code] += int(mask.sum())
                                    key = (var, scl_code)
                                    if key not in scl_stats:
                                        scl_stats[key] = RunningStats(var)
                                    scl_stats[key].update(data, mask)
            finally:
                if scl_vrt is not None:
                    scl_vrt.close()
                if scl_src is not None:
                    scl_src.close()

        for var, stats in scene_stats.items():
            row = {**meta, "file_name": tif_path.name, "variable": var}
            row.update(stats.as_dict())
            variable_rows.append(row)

        for var, stats in flag_stats.items():
            total = stats.total_count
            flag_rows.append(
                {
                    **meta,
                    "file_name": tif_path.name,
                    "variable": var,
                    "qc_total_count": total,
                    "qc_zero_count": stats.zero_count,
                    "qc_zero_pct": stats.zero_count / total if total else None,
                    **{f"flag_{bit}_{name}_count": stats.bit_counts.get(bit, 0) for bit, name in FLAG_BITS.items()},
                    **{
                        f"flag_{bit}_{name}_pct": stats.bit_counts.get(bit, 0) / total if total else None
                        for bit, name in FLAG_BITS.items()
                    },
                    "unique_qc_values_json": json.dumps(dict(sorted(stats.unique_counts.items())), ensure_ascii=False),
                }
            )

        for (var, scl_code), stats in sorted(scl_stats.items()):
            row = {
                **meta,
                "file_name": tif_path.name,
                "variable": var,
                "scl_code": scl_code,
                "scl_class": SCL_CLASSES.get(scl_code, "UNKNOWN"),
            }
            row.update(stats.as_dict())
            scl_rows.append(row)

        scene_row.update(
            {
                f"scl_{code}_{name}_lai_pixels": scl_counts.get(code, 0)
                for code, name in SCL_CLASSES.items()
            }
        )
        scene_rows.append(scene_row)

    return (
        pd.DataFrame(scene_rows),
        pd.DataFrame(variable_rows),
        pd.DataFrame(flag_rows),
        pd.DataFrame(scl_rows),
    )


def build_singleband_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for product_dir in sorted(SINGLE_BAND_DIR.glob("*")):
        if not product_dir.is_dir():
            continue
        for path in sorted(product_dir.glob("*.tif")):
            rows.append(
                {
                    "product": product_dir.name,
                    "file_name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
    return pd.DataFrame(rows)


def build_multiband_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(BIOPHYSICAL_DIR.glob("Biophysical_*.tif")):
        meta = parse_biophysical_name(path.name)
        rows.append(
            {
                **meta,
                "file_name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows)


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_did_snapshot() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    results = DID_DIR / "04_did_results"

    recovery = load_csv_if_exists(results / "recovery_metrics.csv")
    for _, row in recovery.iterrows():
        rows.append(
            {
                "source": "recovery_metrics.csv",
                "outcome": row.get("outcome"),
                "zone_name": row.get("zone_name"),
                "metric": "max_damage_att",
                "value": row.get("max_damage_att"),
                "relative_year": row.get("max_damage_relative_year"),
                "note": "negative means ecological decline after landslide",
            }
        )
        rows.append(
            {
                "source": "recovery_metrics.csv",
                "outcome": row.get("outcome"),
                "zone_name": row.get("zone_name"),
                "metric": "time_to_80pct_recovery",
                "value": row.get("time_to_80pct_recovery"),
                "relative_year": None,
                "note": "years to recover 80 percent of max damage, if available",
            }
        )

    radiation = load_csv_if_exists(results / "radiation_range_summary.csv")
    for _, row in radiation.iterrows():
        rows.append(
            {
                "source": "radiation_range_summary.csv",
                "outcome": row.get("outcome"),
                "zone_name": row.get("significant_zones"),
                "metric": "max_significant_distance_m",
                "value": row.get("max_significant_distance_m"),
                "relative_year": row.get("relative_year"),
                "note": "outermost significant negative ring",
            }
        )

    pretrend = load_csv_if_exists(results / "never_control_pretrend_match.csv")
    if not pretrend.empty:
        for _, row in pretrend.iterrows():
            rows.append(
                {
                    "source": "never_control_pretrend_match.csv",
                    "outcome": row.get("outcome"),
                    "zone_name": row.get("zone_name"),
                    "metric": "matched_controls",
                    "value": row.get("matched_controls", row.get("n_matched_controls", None)),
                    "relative_year": None,
                    "note": "pretrend matched never controls if column is present",
                }
            )

    core_size = load_csv_if_exists(results / "core_size_class_summary.csv")
    for _, row in core_size.iterrows():
        for col in core_size.columns:
            if col in {"size_class", "zone_name"}:
                continue
            rows.append(
                {
                    "source": "core_size_class_summary.csv",
                    "outcome": None,
                    "zone_name": row.get("zone_name"),
                    "metric": f"{row.get('size_class')}_{col}",
                    "value": row.get(col),
                    "relative_year": None,
                    "note": "landslide size class summary",
                }
            )
    return pd.DataFrame(rows)


def write_excel(output_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            clean = sheet_name[:31]
            df.to_excel(writer, sheet_name=clean, index=False)


def write_decision_rules(path: Path) -> None:
    rules = {
        "recommended_first_pass_mask": {
            "keep_scl": [4],
            "drop_scl": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11],
            "interpret_scl_5": "not vegetated; keep only for bare-ground diagnostics, not as reliable vegetation LAI",
            "drop_qc_bits": {
                "bit0": "INPUT_OUT_OF_RANGE",
                "bit3": "OUTPUT_TOO_LOW",
                "bit4": "OUTPUT_TOO_HIGH",
            },
            "review_qc_bits": {
                "bit1": "OUTPUT_THRESHOLDED_TO_MIN_OUTPUT",
                "bit2": "OUTPUT_THRESHOLDED_TO_MAX_OUTPUT",
            },
        },
        "lai_bare_ground_diagnostic": {
            "suspect_rule": "SCL=5 and LAI > 0.8",
            "strong_suspect_rule": "SCL in [1,2,3,8,9,10] and LAI > 0.5",
        },
        "processor_notes": {
            "current_batch_operator": "Biophysical10mOp",
            "current_batch_inputs": "B3, B4, B8 plus sun/view geometry",
            "s2c_warning": "Current batch parser only recognizes S2A/S2B; S2C should be isolated or mapped deliberately after SNAP support is confirmed.",
        },
    }
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    raw_inventory: pd.DataFrame,
    multiband_inventory: pd.DataFrame,
    singleband_inventory: pd.DataFrame,
    missing_lai: pd.DataFrame,
    scene_summary: pd.DataFrame,
    variable_stats: pd.DataFrame,
    flag_counts: pd.DataFrame,
    scl_diag: pd.DataFrame,
    did_snapshot: pd.DataFrame,
    did_files: pd.DataFrame,
    snap_files: pd.DataFrame,
    all_files: pd.DataFrame,
) -> None:
    missing_lai_written = int(missing_lai.get("written_now", pd.Series(dtype=bool)).sum()) if not missing_lai.empty else 0
    matched_scl = int(scene_summary.get("matched_scl_20m", pd.Series(dtype=bool)).sum()) if not scene_summary.empty else 0
    total_scene = len(scene_summary)

    lai_scl5 = pd.DataFrame()
    if not scl_diag.empty:
        lai_scl5 = scl_diag[(scl_diag["variable"] == "LAI") & (scl_diag["scl_code"] == 5)]
    high_scl5 = int(lai_scl5["lai_gt_0_8_count"].fillna(0).sum()) if not lai_scl5.empty else 0
    scl5_total = int(lai_scl5["finite_count"].fillna(0).sum()) if not lai_scl5.empty else 0
    high_scl5_pct = high_scl5 / scl5_total if scl5_total else None

    qc_problem = pd.DataFrame()
    if not flag_counts.empty:
        qc_cols = [col for col in flag_counts.columns if col.endswith("_pct") and col.startswith("flag_")]
        qc_problem = flag_counts[["file_name", "variable", *qc_cols]].copy()

    lines = [
        "# SNAP LAI/FAPAR/FVC quality audit and DID linkage",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- Raw L2A products found: {len(raw_inventory)}",
        f"- SNAP multiband Biophysical GeoTIFFs found: {len(multiband_inventory)}",
        f"- Single-band GeoTIFFs found after audit: {len(singleband_inventory)}",
        f"- DID/EcoService files inventoried: {len(did_files)}",
        f"- SNAP result data files inventoried: {len(snap_files)}",
        f"- Scenes with matched L2A SCL diagnostics: {matched_scl}/{total_scene}",
        f"- Missing LAI single-band rasters written now: {missing_lai_written}",
        "",
        "## Main Findings",
        "",
        "1. The existing batch script uses SNAP `Biophysical10mOp`, so LAI/FAPAR/FVC are estimated from the 10 m green/red/NIR bands plus acquisition geometry.",
        "2. The batch output contains QC bands, but the old single-band export kept only LAI/FAPAR/FVC and did not export QC masks. This audit therefore reads QC directly from the multiband GeoTIFFs.",
        "3. The raw L2A directory currently matches only T49RGH scenes. Two T49RFH Biophysical outputs have no matching raw L2A zip in the provided folder, so SCL-based bare-ground diagnostics are unavailable for those two scenes.",
        "4. The first conservative downstream mask should keep `SCL=4` vegetation and `QC=0`. Use `SCL=5` only as a diagnostic class for bare ground, especially for `LAI > 0.8`.",
    ]
    if high_scl5_pct is not None:
        lines.append(
            f"5. Across matched scenes, `SCL=5` not-vegetated pixels with `LAI > 0.8`: {high_scl5:,} / {scl5_total:,} ({high_scl5_pct:.2%})."
        )
    else:
        lines.append("5. No SCL=5 LAI diagnostic was available, probably because matching SCL was not found.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `l2a_raw_inventory.csv`: raw Sentinel-2 L2A zip/SAFE inventory and SCL availability.",
            "- `snap_multiband_inventory.csv`: SNAP Biophysical multiband outputs.",
            "- `snap_singleband_inventory.csv`: LAI/FAPAR/FVC single-band products after missing LAI repair.",
            "- `biophysical_scene_summary.csv`: file-level scene dimensions and SCL matching.",
            "- `biophysical_variable_stats.csv`: per-scene LAI/FAPAR/FVC distribution statistics.",
            "- `biophysical_qc_flag_counts.csv`: SNAP QC bit counts and percentages.",
            "- `lai_fapar_fvc_by_scl.csv`: per-scene, per-variable SCL grouped statistics where raw SCL is available.",
            "- `did_existing_outputs_inventory.csv`: all current EcoService_DID outputs.",
            "- `did_metric_snapshot.csv`: compact DID result snapshot from existing key CSV files.",
            "- `SNAP_QA_DID_integration_summary.xlsx`: Excel bundle of the key tables.",
            "- `qa_decision_rules.json`: recommended mask and diagnostic rules for the next run.",
            "",
            "## Code Audit Notes",
            "",
            "- `run_lai_batch.py` correctly patches the SNAP graph to `Biophysical10mOp` and fixes the expected band order: `1=LAI, 2=LAIQC, 3=FAPAR, 4=FAPARQC, 5=FVC, 6=FVCQC`.",
            "- The script does not apply SCL masking, cloud/shadow dilation, topographic screening, or QC filtering before single-band export.",
            "- The parser defaults unknown sensors to `S2A_10m`; isolate any future `S2C_MSIL2A` products until the SNAP model choice is explicit.",
            "- For the current mountain landslide workflow, the audit outputs should be used to build a masked LAI/FAPAR/FVC stack before rerunning DID.",
            "",
            "## DID Linkage Snapshot",
            "",
        ]
    )
    if did_snapshot.empty:
        lines.append("- No compact DID snapshot could be built from expected key CSVs.")
    else:
        for _, row in did_snapshot.head(18).iterrows():
            lines.append(
                f"- {row.get('outcome')} / {row.get('zone_name')} / {row.get('metric')}: {row.get('value')} ({row.get('source')})"
            )
    if len(did_snapshot) > 18:
        lines.append(f"- Additional DID snapshot rows: {len(did_snapshot) - 18}; see `did_metric_snapshot.csv`.")

    if not qc_problem.empty:
        lines.extend(["", "## QC Reminder", ""])
        lines.append("Use `biophysical_qc_flag_counts.csv` to remove scenes or pixels with high `INPUT_OUT_OF_RANGE` or `OUTPUT_TOO_HIGH/LOW` rates before interpreting bare-ground LAI.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    raw_inventory, raw_index = build_raw_inventory()
    multiband_inventory = build_multiband_inventory()
    multiband_paths = [Path(path) for path in multiband_inventory["path"].tolist()] if not multiband_inventory.empty else []

    print("Splitting missing LAI single-band rasters if needed...")
    missing_lai = split_missing_lai(multiband_paths)
    singleband_inventory = build_singleband_inventory()

    print("Auditing multiband products and QC flags...")
    scene_summary, variable_stats, flag_counts, scl_diag = audit_biophysical_products(raw_index)

    print("Building DID inventories and snapshots...")
    did_files = pd.DataFrame(file_inventory(DID_DIR, "EcoService_DID"))
    snap_files = pd.DataFrame(file_inventory(SNAP_RESULTS_DIR, "SNAP_Results", skip_parts={"_s2tbx_src", ".git"}))
    all_files = pd.concat([did_files, snap_files], ignore_index=True)
    did_snapshot = build_did_snapshot()

    outputs = {
        "l2a_raw_inventory.csv": raw_inventory,
        "snap_multiband_inventory.csv": multiband_inventory,
        "snap_singleband_inventory.csv": singleband_inventory,
        "missing_lai_singleband_repair.csv": missing_lai,
        "biophysical_scene_summary.csv": scene_summary,
        "biophysical_variable_stats.csv": variable_stats,
        "biophysical_qc_flag_counts.csv": flag_counts,
        "lai_fapar_fvc_by_scl.csv": scl_diag,
        "did_existing_outputs_inventory.csv": did_files,
        "snap_existing_outputs_inventory.csv": snap_files,
        "all_existing_outputs_inventory.csv": all_files,
        "did_metric_snapshot.csv": did_snapshot,
    }
    for name, df in outputs.items():
        df.to_csv(AUDIT_DIR / name, index=False, encoding="utf-8-sig")

    write_decision_rules(AUDIT_DIR / "qa_decision_rules.json")
    write_excel(
        AUDIT_DIR / "SNAP_QA_DID_integration_summary.xlsx",
        {
            "raw_l2a": raw_inventory,
            "snap_multiband": multiband_inventory,
            "singleband": singleband_inventory,
            "scene_summary": scene_summary,
            "variable_stats": variable_stats,
            "qc_flags": flag_counts,
            "by_scl": scl_diag,
            "did_snapshot": did_snapshot,
            "missing_lai_repair": missing_lai,
        },
    )
    write_report(
        AUDIT_DIR / "README_SNAP_LAI_QA_DID.md",
        raw_inventory,
        multiband_inventory,
        singleband_inventory,
        missing_lai,
        scene_summary,
        variable_stats,
        flag_counts,
        scl_diag,
        did_snapshot,
        did_files,
        snap_files,
        all_files,
    )

    print(f"Done. Audit outputs written to {AUDIT_DIR}")


if __name__ == "__main__":
    main()
