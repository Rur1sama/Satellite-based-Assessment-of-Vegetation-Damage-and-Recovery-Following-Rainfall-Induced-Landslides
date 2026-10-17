"""
==============================================================================
Project       : Slide_dect
Module Name   : Optical_feature.py
File Path     : 代码/机器学习训练/训练主程序/utils/Optical_feature.py
File Created  : 2026-05-28 22:08:57
Description   : 光学特征提取--特征提取模块
Lines         : 61
==============================================================================
"""

from utils.__init__ import * # 初始化参数配置、模型参数
from utils.base_utils import smart_align


def safe_ndvi(nir, red):
    '''
    安全计算NDVI，防止除以零或无效值
    
    :param nir: 近红外波段np数组
    :param red: 红波段np数组
    :return: ndvi数组
    '''
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi[(nir + red) <= 0] = np.nan
    return ndvi

def safe_ndwi(nir,green):
    '''
    安全计算NDWI，防止除以零或无效值

    :param nir: 近红外波段 np数组
    :param green: 绿波段 np数组
    :return: ndwi np数组
    '''
    ndwi = (green- nir) / ( green + nir  + 1e-8)
    ndwi[(nir + green) <= 0] = np.nan
    return ndwi

def build_optical_features(src, pre_path, profile):
    '''
    构建光学指数特征。

    :param src:             灾后光学影像
    :param pre_path:        灾后光学影像对应灾前光学影像路径
    :param profile:         栅格模板
    :return: ndvi_post:     灾后ndvi
    \n       delta_ndvi:    ndvi变化     
    \n       ndwi_post:     灾后ndwi
    
    '''
    red_post = src.read(CONFIG["BANDS"]["RED"]).astype(np.float32)
    nir_post = src.read(CONFIG["BANDS"]["NIR"]).astype(np.float32)
    green_post = src.read(CONFIG["BANDS"]["Green"]).astype(np.float32)
    ndvi_post = safe_ndvi(nir_post, red_post)
    delta_ndvi = safe_ndvi(
        smart_align(pre_path, profile, CONFIG["BANDS"]["NIR"]),
        smart_align(pre_path, profile, CONFIG["BANDS"]["RED"])
    ) - ndvi_post
    ndwi_post = safe_ndwi(nir_post, green_post)
    return ndvi_post, delta_ndvi, ndwi_post