"""
==============================================================================
Project       : Slide_dect
Module Name   : head.py
Description   : Central imports and runtime settings for ecosystem DID analysis.
==============================================================================
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from rasterio.warp import Resampling, reproject
from scipy import stats
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from tqdm import tqdm

warnings.filterwarnings("ignore")
np.seterr(divide="ignore", invalid="ignore", over="ignore")

