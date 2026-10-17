"""
==============================================================================
Project       : Slide_dect
Module Name   : Terrain_feature.py
File Path     : 代码/机器学习训练/训练主程序/utils/Terrain_feature.py
File Created  : 2026-05-28 22:06:47
Description   : 地形特征提取--特征提取模块
Lines         : 107
==============================================================================
"""

from utils.head import * # 导入所有头文件
from utils.__init__ import * 
from utils.base_utils import smart_align
from utils.base_utils import save_raster_if_missing

def calculate_curvature(dem, res):
    dy, dx = np.gradient(dem, res)
    d2y, _ = np.gradient(dy, res)
    _, d2x = np.gradient(dx, res)
    return d2x + d2y


def calculate_twi_pysheds(dem, res, ref_profile, dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    '''
    基于 pysheds 官方流程计算 TWI。 
    '''
    dem = np.asarray(dem, dtype=np.float32)
    valid_mask = np.isfinite(dem)
    if not np.any(valid_mask):
        return np.full_like(dem, np.nan, dtype=np.float32)
    nodata = -9999.0
    dem_write = np.where(valid_mask, dem, nodata).astype(np.float32)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        tmp_profile = ref_profile.copy()
        tmp_profile.update(driver="GTiff", count=1, dtype="float32", nodata=nodata)
        with rasterio.open(tmp_path, "w", **tmp_profile) as dst:
            dst.write(dem_write, 1)
        grid = Grid.from_raster(tmp_path)
        dem_raster = grid.read_raster(tmp_path, nodata=nodata)
        pit_filled = grid.fill_pits(dem_raster)
        flooded = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(flooded)
        fdir = grid.flowdir(inflated, dirmap=dirmap)
        acc = grid.accumulation(fdir, dirmap=dirmap)
        dy, dx = np.gradient(dem, res)
        slope_tan = np.maximum(np.sqrt(dx**2 + dy**2), 1e-3)
        acc_arr = np.asarray(acc, dtype=np.float32)
        twi = np.log(((acc_arr + 1.0) * res) / slope_tan)
        twi[~valid_mask] = np.nan
        twi[~np.isfinite(twi)] = np.nan
        return twi.astype(np.float32)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def build_terrain_features(profile):
    '''
    构建地形因子，并保存坡度与坡向栅格。

    :param profile: rasterio 生成的raster模板，包含：
        'driver': 'GTiff',      # 文件格式\n
        'width': 1000,          # 宽度（像素）\n
        'height': 800,          # 高度（像素）\n
        'count': 3,             # 波段数\n
        'dtype': 'uint8',       # 数据类型\n
        'crs': 'EPSG:32649',     # 坐标参考系\n
        'transform': affine,    # 仿射变换参数\n
        'compress': 'lzw',      # 压缩方式（可选）\n
        'tiled': True,          # 是否分块（可选）\n
        'blockxsize': 256,      # 块大小（可选）\n
        'blockysize': 256       # 块大小（可选）\n

    :return:  dem       (数字高程模型)
    \n        res       (参考坐标系)
    \n        curvature (曲率)
    \n        aspect    (坡向) 
    \n        aspect_ns (南北坡向) 
    \n        aspect_ew (东西坡向) 
    \n        slope     (坡度)
    \n        twi       (地形水文指数)
    '''
    dem = smart_align(CONFIG["DEM_PATH"], profile)
    res = abs(profile['transform'][0])
    dy, dx = np.gradient(dem, res)
    curvature = calculate_curvature(dem, res)
    twi = calculate_twi_pysheds(dem,res,profile)
    aspect = ((np.degrees(np.arctan2(-dx, dy)) + 360.0) % 360.0)
    aspect_rad = np.radians(aspect)
    aspect_ns = np.cos(aspect_rad)
    aspect_ew = np.sin(aspect_rad)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # 保存坡度数据和坡向数据，若已有即跳过保存
    slope_out_path = os.path.join(CONFIG["OUTPUT_DIR"], f"Slope.tif")
    aspect_out_path = os.path.join(CONFIG["OUTPUT_DIR"], f"Aspect.tif")
    save_raster_if_missing(slope, slope_out_path, profile)
    save_raster_if_missing(aspect, aspect_out_path, profile)

    return dem, res, curvature, aspect, aspect_ns, aspect_ew, slope, twi