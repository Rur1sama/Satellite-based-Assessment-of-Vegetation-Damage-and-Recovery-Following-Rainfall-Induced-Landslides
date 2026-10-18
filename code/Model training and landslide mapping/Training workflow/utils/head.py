"""
==============================================================================
Project       : Slide_dect
Module Name   : head.py
File Path     : code/Model training and landslide mapping/Training workflow/utils/head.py
File Created  : 2026-05-28 21:52:49
Description   : 运行环境依赖模块
Lines         : 57
==============================================================================
"""

import os
import glob
import warnings
import datetime
import tempfile
import joblib
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from scipy.ndimage import generic_filter
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import seaborn as sns

# 引入 4 大机器学习核心框架
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                             ConfusionMatrixDisplay, roc_curve, auc, 
                             precision_recall_curve)
from pyproj import Transformer
from pysheds.grid import Grid

# 屏蔽预料内的数学计算警告 (如NDVI计算中的空值或除零)
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore', over='ignore')

from tqdm import tqdm    # 引入tqdm进度条库

# SHAP解释库
from joblib import Parallel, delayed
import shap
import gc

from rasterio.features import shapes # 显式引入栅格转矢量工具
from shapely.geometry import shape