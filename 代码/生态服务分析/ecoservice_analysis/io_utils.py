from __future__ import annotations

import json
import logging
import glob
import re
from pathlib import Path
from typing import Any


def setup_logger(log_path: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("ecoservice_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg, config_path.parent


def resolve_path(value: str | Path | None, base_dir: str | Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(base_dir).resolve() / path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_to_csv(df: Any, path: str | Path, logger: logging.Logger | None = None, **kwargs: Any) -> Path:
    """
    安全写出 CSV。

    Windows 下如果目标 CSV 正被 Excel 或预览器占用，直接覆盖会触发 PermissionError。
    这种情况下写成 *_latest.csv，让当前流程可以继续跑完。
    """
    target = ensure_parent(path)
    try:
        df.to_csv(target, **kwargs)
        return target
    except PermissionError:
        fallback = target.with_name(f"{target.stem}_latest{target.suffix}")
        df.to_csv(fallback, **kwargs)
        if logger is not None:
            logger.warning("CSV is locked, wrote fallback file instead: %s", fallback)
        return fallback


def driver_from_path(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    if suffix == ".gpkg":
        return "GPKG"
    if suffix == ".shp":
        return "ESRI Shapefile"
    if suffix in {".geojson", ".json"}:
        return "GeoJSON"
    return None


def expand_yearly_rasters(
    outcome_name: str,
    outcome_cfg: dict[str, Any],
    base_dir: str | Path,
    skip_missing: bool,
    logger: logging.Logger | None = None,
) -> dict[int, list[Path]]:
    if not outcome_cfg.get("enabled", True):
        return {}

    rasters: dict[int, list[Path]] = {}
    if "rasters" in outcome_cfg:
        raw_rasters = outcome_cfg["rasters"]
        if isinstance(raw_rasters, dict):
            for year, path in raw_rasters.items():
                paths = path if isinstance(path, list) else [path]
                rasters[int(year)] = [
                    p for p in (resolve_path(item, base_dir) for item in paths) if p is not None
                ]
        elif isinstance(raw_rasters, list):
            for item in raw_rasters:
                year = int(item["year"])
                rasters.setdefault(year, [])
                path = resolve_path(item["path"], base_dir)
                if path is not None:
                    rasters[year].append(path)
        else:
            raise ValueError(f"{outcome_name}.rasters must be a dict or list")
    elif "glob_pattern" in outcome_cfg:
        years = [int(y) for y in outcome_cfg.get("years", [])]
        pattern_template = outcome_cfg["glob_pattern"]
        year_regex = outcome_cfg.get("year_regex")
        compiled = re.compile(year_regex) if year_regex else None

        if "{year}" in pattern_template:
            for year in years:
                pattern = str(resolve_path(pattern_template.format(year=year), base_dir))
                matched = sorted(Path(p) for p in glob.glob(pattern))
                rasters[year] = matched
        else:
            pattern = str(resolve_path(pattern_template, base_dir))
            for matched in sorted(Path(p) for p in glob.glob(pattern)):
                year: int | None = None
                if compiled:
                    match = compiled.search(matched.name)
                    if match:
                        year = int(match.group(1))
                if year is None:
                    year_match = re.search(r"(20\d{2})", matched.name)
                    if year_match:
                        year = int(year_match.group(1))
                if year is None:
                    if logger is not None:
                        logger.warning("Cannot infer year for %s raster: %s", outcome_name, matched)
                    continue
                if years and year not in years:
                    continue
                rasters.setdefault(year, []).append(matched)
    elif "path_template" in outcome_cfg:
        years = outcome_cfg.get("years", [])
        template = outcome_cfg["path_template"]
        for year in years:
            path = resolve_path(template.format(year=int(year)), base_dir)
            rasters[int(year)] = [path] if path is not None else []
    else:
        raise ValueError(f"{outcome_name} needs rasters, glob_pattern, or path_template")

    existing: dict[int, list[Path]] = {}
    for year, paths in rasters.items():
        year_paths: list[Path] = []
        for path in paths:
            if path.exists():
                year_paths.append(path)
            elif skip_missing:
                if logger is not None:
                    logger.warning("Skip missing %s raster for %s: %s", outcome_name, year, path)
            else:
                raise FileNotFoundError(f"Missing {outcome_name} raster for {year}: {path}")
        if year_paths:
            existing[year] = year_paths
        elif not skip_missing:
            raise FileNotFoundError(f"No {outcome_name} raster found for {year}")

    return existing


def outcome_specs_from_config(
    cfg: dict[str, Any],
    base_dir: str | Path,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, Any]]:
    skip_missing = bool(cfg.get("skip_missing_rasters", True))
    specs: dict[str, dict[str, Any]] = {}
    for name, outcome_cfg in cfg.get("outcomes", {}).items():
        if not outcome_cfg.get("enabled", True):
            continue
        rasters = expand_yearly_rasters(name, outcome_cfg, base_dir, skip_missing, logger)
        if not rasters:
            if logger is not None:
                logger.warning("No rasters found for outcome %s; it will be skipped.", name)
            continue
        spec = dict(outcome_cfg)
        spec["rasters"] = rasters
        specs[name] = spec
    return specs


def mask_specs_from_config(
    cfg: dict[str, Any],
    base_dir: str | Path,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    landslide_cfg = cfg["landslide"]
    if "glob_pattern" in landslide_cfg:
        pattern_template = landslide_cfg["glob_pattern"]
        for year_pair in landslide_cfg.get("year_pairs", []):
            year_pre, year_post = str(year_pair).split("-")
            pattern = str(
                resolve_path(
                    pattern_template.format(year_pre=year_pre, year_post=year_post, year_pair=year_pair),
                    base_dir,
                )
            )
            matched = sorted(Path(p) for p in glob.glob(pattern))
            if not matched:
                if cfg.get("skip_missing_rasters", True):
                    if logger is not None:
                        logger.warning("Skip missing landslide mask pattern: %s", pattern)
                    continue
                raise FileNotFoundError(f"Missing landslide mask pattern: {pattern}")
            for path in matched:
                specs.append(
                    {
                        "period": f"20{year_pre}-20{year_post}",
                        "event_year": int(f"20{year_post}"),
                        "path": path,
                    }
                )
        return specs

    for item in cfg["landslide"]["masks"]:
        path = resolve_path(item["path"], base_dir)
        if path is None:
            continue
        if not path.exists():
            if cfg.get("skip_missing_rasters", True):
                if logger is not None:
                    logger.warning("Skip missing landslide mask: %s", path)
                continue
            raise FileNotFoundError(f"Missing landslide mask: {path}")
        spec = dict(item)
        spec["path"] = path
        specs.append(spec)
    return specs
