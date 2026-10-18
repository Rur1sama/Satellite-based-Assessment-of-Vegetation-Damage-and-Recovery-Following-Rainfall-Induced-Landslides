"""
==============================================================================
Project       : Slide_dect
Module Name   : config.py
Description   : Default paths and parameters for landslide ecosystem DID.
==============================================================================
"""

from __future__ import annotations

from pathlib import Path

from .head import os


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ECO_INPUT = PROJECT_ROOT / "Input" / "EcologicalAnalysis"
ECO_OUTPUT = PROJECT_ROOT / "Output" / "EcologicalAnalysis"

CONFIG = {
    "PROJECT_ROOT": str(PROJECT_ROOT),
    "OUTPUT_DIR": str(ECO_OUTPUT),
    "STUDY_AREA_PATH": str(ECO_INPUT / "ExampleData" / "StudyArea" / "study_area_boundary.shp"),
    "LANDSLIDE_MASK_PATTERN": str(PROJECT_ROOT / "Output" / "LandslideProducts" / "Data" / "annual_probability_maps" / "landslide_probability_{year_pre}_{year_post}.tif"),
    "LAI_PATTERN": str(PROJECT_ROOT / "Output" / "EcologicalProducts" / "LAI" / "LAI_{year}????_*.tif"),
    "GPP_PATTERN": str(PROJECT_ROOT / "Output" / "EcologicalProducts" / "GPP" / "GPP_{year}????_existing_epsilon_kgC_m2_day.tif"),
    "YEAR_PAIRS": ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"],
    "YEARS": list(range(2018, 2026)),
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

