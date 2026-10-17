"""
==============================================================================
Project       : Slide_dect
Module Name   : base_utils.py
File Path     : 代码/机器学习训练/训练主程序/utils/base_utils.py
File Created  : 2026-06-01 14:16:48
Description   : 基本工具--数据预处理模块
Lines         : 81
==============================================================================
"""

from utils.head import * # 导入所有头文件
from utils.__init__ import *

def smart_align(src_path, ref_profile, band_idx=1):
    '''
    将栅格影像根据参考影像进行重投影并对齐，双线性插值法重采样
    '''
    with rasterio.open(src_path) as src:
        dest = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)
        reproject(rasterio.band(src, band_idx), dest, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'], resampling=Resampling.bilinear)
        if src.nodata is not None:
            dest[dest == np.float32(src.nodata)] = np.nan
        return dest


def get_year_input_paths(year_pair):
    '''
    获取当前年份对需要使用的光学影像与标签路径。

    :param year_pair: 指定年份对
    
    :return: pre_year       所有灾前年份光学影像的字符串数组 
    \n       post_year      所有灾后年份光学影像的字符串数组
    \n       pre_paths[0]   当前年份对的灾前光学影像路径
    \n       post_paths[0]  当前年份对的灾后光学影像路径
    \n       label_paths[0] 当前年份对的标签路径
    '''
    pre_year, post_year = "20" + year_pair.split("-")[0], "20" + year_pair.split("-")[1]
    pre_paths = glob.glob(os.path.join(CONFIG["OPTICAL_DIR"], f"SR2_{pre_year}*", "*.tif"))
    post_paths = glob.glob(os.path.join(CONFIG["OPTICAL_DIR"], f"SR2_{post_year}*", "*.tif"))
    label_paths = glob.glob(os.path.join(CONFIG["LABEL_ROOT"], year_pair, "*.shp"))

    if not pre_paths:
        raise FileNotFoundError(f"未找到灾前年份光学影像: {pre_year}")
    if not post_paths:
        raise FileNotFoundError(f"未找到灾后年份光学影像: {post_year}")
    if not label_paths:
        raise FileNotFoundError(f"未找到年份对标签文件: {year_pair}")

    return pre_year, post_year, pre_paths[0], post_paths[0], label_paths[0]

def save_raster_if_missing(array, out_path, profile):
    '''
    若目标栅格不存在，则保存计算结果。
    '''
    if os.path.exists(out_path):
        return

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array.astype(np.float32), 1)

def rasterize_year_labels(label_path, profile, shape):
    '''
    将当前年份对的矢量标签栅格化。
    '''
    h, w = shape
    gdf = gpd.read_file(label_path).to_crs(profile['crs'])
    label_shapes = [
        (geom, int(value))
        for geom, value in zip(gdf.geometry, gdf['LsBg'])
        if geom and not pd.isna(value)
    ]
    return rasterize(
        label_shapes,
        out_shape=(h, w),
        transform=profile['transform'],
        fill=0,
        dtype=np.int16
    )
