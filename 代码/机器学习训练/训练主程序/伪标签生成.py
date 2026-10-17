"""
==============================================================================
Project       : Slide_dect
Module Name   : 伪标签生成.py
File Path     : 代码/机器学习训练/训练主程序/伪标签生成.py
File Created  : 2026-03-01 14:15:49
Description   : Description   : 伪标签生成 - 数据预处理模块
Lines         : 123
==============================================================================
"""
from utils.head import * # 导入所有基础库 (os, numpy, rasterio, cv2等)
from utils.__init__ import * # 导入 CONFIG 配置和 log 记录功能
from utils.base_utils import * # 导入 smart_align, save_raster_if_missing
from utils.Optical_feature import build_optical_features
from utils.Terrain_feature import build_terrain_features

# =====================================================================
# 1. 局部阈值与参数配置 (结合你的启发式阈值库)
# =====================================================================
THRESHOLDS = {
    "NDVI_PRE_MIN": 0.2,    # 灾前植被覆盖基准：必须有绿度（剔除原本就是裸土或岩石的区域）
    "DELTA_NDVI_MIN": 0.15, # 植被衰减阈值：灾后 NDVI 下降必须显著
    "SLOPE_MIN": 5.0,       # 地形坡度阈值：滑坡通常发生在有一定坡度的地形上 (度)
    "NDWI_POST_MAX": 0      # 水体剔除阈值：灾后 NDWI 大于此值视为水体，坚决剔除
}

def generate_pseudo_label_mask(year_pair):
    """
    读取指定年份对的数据，计算光学与地形特征，通过多准则交集及水体掩膜过滤，生成潜在滑坡二值掩膜。
    """
    log(f"开始执行 [{year_pair}] 伪标签生成 ...")
    
    # 1.1 获取输入路径 
    try:
        pre_year, post_year, pre_path, post_path, _ = get_year_input_paths(year_pair)
    except FileNotFoundError as e:
        log(f"  [Error] 路径获取失败: {e}")
        return

    # 1.2 打开灾后影像作为空间参考
    with rasterio.open(post_path) as src:
        ref_profile = src.profile.copy()
        ref_profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        
        log(f"  [1/3] 开始提取光谱指数特征...")
        # 调用你 Optical_feature 中的专业级安全计算函数，获取灾后NDVI、变化量及NDWI
        ndvi_post, delta_ndvi, ndwi_post = build_optical_features(src, pre_path, ref_profile)
        # 恢复灾前 NDVI 用于植被基准过滤
        ndvi_pre = ndvi_post + delta_ndvi

        log(f"  [2/3] 开始构建地形坡度特征...")
        # 调用你 Terrain_feature 中的地形管道计算坡度 (自动对齐 DEM)
        _, _, _, _, _, _, slope, _ = build_terrain_features(ref_profile)
        
        log(f"  [3/3] 执行关学掩膜与地形掩膜剔除...")
        # 准则 A: 灾前必须有植被复盖
        cond_pre_veg = (ndvi_pre >= THRESHOLDS["NDVI_PRE_MIN"])
        # 准则 B: 灾后植被发生显著衰减
        cond_veg_loss = (delta_ndvi >= THRESHOLDS["DELTA_NDVI_MIN"])
        # 准则 C: 必须具备滑坡发生的地形坡度
        cond_slope = (slope >= THRESHOLDS["SLOPE_MIN"]) 
        # 核心逻辑：初步滑坡候选区 (A ∩ B ∩ C)
        candidate_mask = cond_pre_veg & cond_veg_loss & cond_slope
        # 水体剔除逻辑 (利用灾后 NDWI 指数进行强力过滤，防止河流、湖泊或季节性积水误判)
        water_mask = (ndwi_post > THRESHOLDS["NDWI_POST_MAX"])
        # 最终滑坡二值掩膜：属于候选区且排除掉水体区域
        final_mask = candidate_mask & (~water_mask)
        # 数据清洗：剔除计算中产生的 NaN 干扰值
        invalid_nan = np.isnan(ndvi_pre) | np.isnan(delta_ndvi) | np.isnan(slope) | np.isnan(ndwi_post)
        final_mask[invalid_nan] = False
        # 转换为无符号8位整型 (0: 背景, 1: 潜在滑坡)
        binary_label = final_mask.astype(np.uint8)
        # 统计初筛统计数据
        total_pixels = binary_label.size
        landslide_pixels = np.sum(binary_label == 1)
        water_pixels = np.sum(water_mask)
        log(f"    > 栅格总像素数: {total_pixels}")
        log(f"    > 剔除水体像素数: {water_pixels}")
        log(f"    > 筛选出的潜在滑坡像素数: {landslide_pixels} ({landslide_pixels/total_pixels:.4%})")

        # 1.4 结果保存
        output_dir = os.path.join(CONFIG["OUTPUT_DIR"], "Pseudo_Labels")
        os.makedirs(output_dir, exist_ok=True)
        out_tif_path = os.path.join(output_dir, f"Pseudo_Mask_{year_pair}.tif")
        ref_profile.update(dtype=rasterio.uint8, nodata=0)
        with rasterio.open(out_tif_path, 'w', **ref_profile) as dst:
            dst.write(binary_label, 1)
        log(f"  [Success] 年份对 {year_pair} 的潜在滑坡掩膜已成功输出至: {out_tif_path}")
        # 将tif保存为矢量
        geoms = []
        geom_records = []
        # 使用 rasterio.features.shapes 提取值为 1 的多边形
        mask_for_shapes = (binary_label == 1)
        shape_generator = shapes(binary_label, mask=mask_for_shapes, transform=ref_profile['transform'])
        for i, (geom, val) in enumerate(shape_generator):
            geoms.append(shape(geom))  # 转换为 shapely 几何对象
            geom_records.append({
                "Poly_ID": i + 1,       # 斑块唯一标识
                "Class": int(val),      # 类别：1
                "Year_Pair": year_pair  # 时段标签
            })
        if geoms:
            # 组装 GeoDataFrame
            gdf = gpd.GeoDataFrame(geom_records, geometry=geoms, crs=ref_profile['crs'])
            # 定义矢量输出路径
            out_shp_path = os.path.join(output_dir, f"Pseudo_Shapefile_{year_pair}.shp")
            # 保存为 ESRI Shapefile
            gdf.to_file(out_shp_path, driver="ESRI Shapefile", encoding='utf-8')
            log(f"  [Success] 年份对 {year_pair} 的潜在滑坡矢量已成功导出至: {out_shp_path}")
        else:
            log(f"  [Info] 年份对 {year_pair} 未筛选出有效滑坡斑块，跳过矢量导出。")

if __name__ == "__main__":
    log("="*60)
    log("滑坡伪标签生成流程！启动！ ")
    log("="*60)
    
    # 自动循环处理你在 CONFIG 中定义的所有研究时段
    for yp in CONFIG["YEAR_PAIRS"]:
        generate_pseudo_label_mask(yp)
        log("-"*60)
        
    log("所有年份对的潜在滑坡掩膜构建任务已全部完成！")