"""
==============================================================================
Project       : Slide_dect
Module Name   : config.py
Description   : Default paths and parameters for landslide ecosystem DID.
==============================================================================
"""

from __future__ import annotations

from .head import os


CONFIG = {
    "PROJECT_ROOT": r"E:\python_base\Slide_dect",
    "OUTPUT_DIR": r"E:\python_base\Slide_dect\Output\EcoService_DID",
    "STUDY_AREA_PATH": r"E:\python_base\Slide_dect\Input\Jiangwan\Jiangwan.shp",
    "LANDSLIDE_MASK_PATTERN": r"E:\python_base\Slide_dect\Output\ML_Results\Ensemble_Final\Ensemble_Final_{year_pre}-{year_post}.tif",
    "LAI_PATTERN": r"E:\Raster_data\LAI\Sentinel-2\Jiangwan\tile\Biophysical_{year}????_*_*_*_S2*_10m.tif",
    "GPP_PATTERN": r"E:\Raster_data\MODIS\南岭\MOD17\MOD17A3HGF\MCD17A3H_v061_{year}_*.tif",
    "YEAR_PAIRS": ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"],
    "YEARS": list(range(2018, 2026)),
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

