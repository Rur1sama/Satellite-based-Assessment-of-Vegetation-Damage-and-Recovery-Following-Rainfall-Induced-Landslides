import os
import glob
import re
import cv2  
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from datetime import datetime
from tqdm import tqdm
import warnings

# 忽略 geopandas 保存 shapefile 时的字段名长度警告 (Shapefile 字段名限制为10个字符)
warnings.filterwarnings("ignore")

# =====================================================================
# 1. 全局配置与阈值库 (Configuration & Thresholds)
# =====================================================================
CONFIG = {
    # ---------------- 基础数据路径 ----------------
    "S2_PRE_EVENT": r"E:/python_base/Slide_dect/Input/光学栅格/SR2_20231120\Sentinel2_20231120T030021_20231120T030421_T49RGH.tif",                # 灾前光学影像
    "S2_POST_EVENT": r"E:/python_base/Slide_dect/Input/光学栅格/SR2_20241129/SR2_20241129T0300.tif",                # 灾后光学影像"
    "RAW_DEM": r"E:/python_base/Slide_dect/Input/高程数据/ALOS_PALSAR_DEM.tif",             # 原始数字高程模型 (DEM)
    "ROI_SHP": r"E:/python_base/Slide_dect/Input/Jiangwan/Jiangwan.shp",                    # 研究区矢量边界
    
    # ---------------- 时序数据文件夹 ----------------
    "SAR_DIR": r"E:/python_base/Slide_dect/Input/SAR_Time_Series/",  # SAR雷达时序文件夹
    "GPM_DIR": r"E:/python_base/Slide_dect/Input/GPM_Time_Series/",  # GPM降水时序文件夹
    
    # ---------------- 结果输出路径 (多维产物) ----------------
    "OUT_DIR": r"E:/python_base/Slide_dect/Output/Masks/",
    "OUTPUT_MASK": r"E:/python_base/Slide_dect/Output/Masks/1_Final_Optical_Mask.tif",   # 最终的光学滑坡掩膜
    "OUTPUT_MASK_VECTOR": r"E:/python_base/Slide_dect/Output/Vector/Optical_Mask.shp",   # 最终的光学滑坡掩膜shp
    "OUTPUT_SLOPE": r"E:/python_base/Slide_dect/Output/Masks/0_Generated_Slope.tif",     # 物理坡度模型
    "OUTPUT_DELTA_NDVI": r"E:/python_base/Slide_dect/Output/Masks/0_Delta_NDVI.tif",     # 【新增】NDVI植被锐减连续变化图层
    "OUTPUT_VECTOR": r"E:/python_base/Slide_dect/Output/Vector/Landslide_Instances.shp", # 最终附带全生命周期属性的矢量图层
    
    # ---------------- 机理参数阈值 ----------------
    "GREEN_BAND_IDX": 3,         # 绿波段索引 (用于计算NDWI排除水体)
    "RED_BAND_IDX": 4,           # 红波段索引 (用于计算NDVI)
    "NIR_BAND_IDX": 8,           # 近红外波段索引 (用于计算NDVI)
    "NDVI_DROP_THRESH": 0.15,    # 植被衰减阈值 (大于此值认为是疑似滑坡)
    "SLOPE_THRESH": 5.0,         # 坡度阈值 (度，大于此值才符合重力滑坡物理机理，过滤平原农田)
    "NDWI_WATER_THRESH": 0.0     # 水体阈值 (大于此值认定为水系/洪涝，免除误判)
}

# =====================================================================
# 2. 空间工具库 (Spatial & Terrain Engines)
# =====================================================================
def align_raster(src_path, ref_meta, resampling_method=Resampling.bilinear):
    """
    将任意输入的遥感影像，强制重采样并严格投影对齐到参考网格 (Reference Grid) 上。
    这是解决多源遥感数据 (分辨率不同、坐标系不同) 空间错位的核心底层函数。
    """
    with rasterio.open(src_path) as src:
        # 保持原始影像的波段数 (3D 张量结构: C, H, W)，防止维度塌陷
        aligned_data = np.empty((src.count, ref_meta['height'], ref_meta['width']), dtype=np.float32)
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=aligned_data[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta['transform'],
                dst_crs=ref_meta['crs'],
                resampling=resampling_method
            )
    return aligned_data 

def calculate_slope_from_dem(dem_data, transform):
    """
    利用高程数据 (DEM) 通过 Numpy 的空间偏导数算法计算物理坡度 (单位: 度)。
    """
    dem_data = np.where(dem_data < -1000, np.nan, dem_data) # 过滤异常极值
    pixel_width, pixel_height = abs(transform[0]), abs(transform[4])
    dy, dx = np.gradient(dem_data, pixel_height, pixel_width) # 计算行列梯度
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))             # 坡度计算公式
    return np.nan_to_num(np.degrees(slope_rad), nan=0.0).astype(np.float32)

def extract_date_from_filename(filename):
    """
    利用正则表达式，智能捕捉影像文件名中的 8位 YYYYMMDD 日期标识。
    """
    match = re.search(r'(?:19|20)\d{6}', filename)
    if match: return datetime.strptime(match.group(), '%Y%m%d')
    raise ValueError(f"无法解析日期: {filename}")

def cv_clean_mask(raw_mask):
    """
    基于 OpenCV 的形态学连通域分析引擎。
    用于抹除面积过小的孤立噪点，以及长宽比极度失调的卫星拼接缝 (伪影)。
    """
    mask_for_cv = raw_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_for_cv, connectivity=8)
    clean_mask = np.zeros_like(mask_for_cv, dtype=bool)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 15: continue  # 剔除极小面积噪点
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        if aspect_ratio > 10.0 and max(w, h) > 100: continue # 剔除极度细长的直线边界
        clean_mask[labels == i] = True
    return clean_mask, labels

# =====================================================================
# 3. 核心业务逻辑模块化 (Business Logic Modules)
# =====================================================================
def prepare_spatial_data():
    """
    模块 1：数据基准建立与对齐。
    以灾后影像 + Shapefile 为绝对基准，把所有光学波段和 DEM 都裁剪并拉扯到统一网格上。
    """
    print("[1/3] 正在加载研究区矢量，并生成裁剪后的绝对基准网格...")
    with rasterio.open(CONFIG["S2_POST_EVENT"]) as post_src:
        roi_gdf = gpd.read_file(CONFIG["ROI_SHP"]).to_crs(post_src.crs)
        post_data, out_transform = mask(post_src, [geom for geom in roi_gdf.geometry], crop=True)
        ref_meta = post_src.meta.copy()
        ref_meta.update({"height": post_data.shape[1], "width": post_data.shape[2], "transform": out_transform})

    print("[1.5/3] 正在对齐灾前光学影像与地形坡度...")
    pre_data = align_raster(CONFIG["S2_PRE_EVENT"], ref_meta)
    
    print("      -> [地形引擎] 正在计算地表偏导数与坡度矩阵...")
    slope_data = calculate_slope_from_dem(align_raster(CONFIG["RAW_DEM"], ref_meta)[0], ref_meta['transform'])

    # 建立输出目录
    os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["OUTPUT_VECTOR"]), exist_ok=True)
    
    # 顺手保存物理坡度模型供核验
    slope_meta = ref_meta.copy()
    slope_meta.update({"count": 1, "dtype": rasterio.float32, "nodata": 0})
    with rasterio.open(CONFIG["OUTPUT_SLOPE"], 'w', **slope_meta) as dst:
        dst.write(slope_data, 1)

    # 提取需要的特定波段
    bands = {
        "pre_g": pre_data[CONFIG["GREEN_BAND_IDX"]-1].astype(np.float32),
        "pre_r": pre_data[CONFIG["RED_BAND_IDX"]-1].astype(np.float32),
        "pre_n": pre_data[CONFIG["NIR_BAND_IDX"]-1].astype(np.float32),
        "post_g": post_data[CONFIG["GREEN_BAND_IDX"]-1].astype(np.float32),
        "post_r": post_data[CONFIG["RED_BAND_IDX"]-1].astype(np.float32),
        "post_n": post_data[CONFIG["NIR_BAND_IDX"]-1].astype(np.float32),
        "slope": slope_data
    }
    return ref_meta, bands


def generate_optical_mask(ref_meta, bands):
    """
    模块 2：光学机理综合漏斗。
    综合植被锐减 (Delta NDVI) + 陡坡约束 (Slope) + 水体免疫 (NDWI) 锁定真实滑坡面。
    """
    print("\n[2/3] 执行光学面萃取：计算 ΔNDVI 并结合 NDWI 水体免疫约束...")
    
    # 获取有效观测区域 (剔除卫星边缘的纯黑无数据区)
    valid_mask = (bands["pre_g"] > 0) & (bands["pre_r"] > 0) & (bands["pre_n"] > 0) & \
                 (bands["post_g"] > 0) & (bands["post_r"] > 0) & (bands["post_n"] > 0)
    
    # 1. 计算灾前、灾后 NDVI 以及植被变化差值
    ndvi_pre = (bands["pre_n"] - bands["pre_r"]) / (bands["pre_n"] + bands["pre_r"] + 1e-8)
    ndvi_post = (bands["post_n"] - bands["post_r"]) / (bands["post_n"] + bands["post_r"] + 1e-8)
    delta_ndvi = ndvi_pre - ndvi_post
    
    # ==========================================================
    # 🌟 核心需求输出：保存 Delta NDVI 连续值变化图层供可视化
    # ==========================================================
    print("      -> 正在导出 ΔNDVI (植被锐减量) 浮点型栅格图层...")
    ndvi_meta = ref_meta.copy()
    ndvi_meta.update({"count": 1, "dtype": rasterio.float32, "nodata": -9999.0})
    # 用 np.where 保证只有有效像素有 NDVI 值，其余都是背景黑边
    out_delta_ndvi = np.where(valid_mask, delta_ndvi, -9999.0).astype(np.float32)
    with rasterio.open(CONFIG["OUTPUT_DELTA_NDVI"], 'w', **ndvi_meta) as dst:
        dst.write(out_delta_ndvi, 1)

    # 2. 计算灾前、灾后 NDWI 水体指数
    ndwi_pre = (bands["pre_g"] - bands["pre_n"]) / (bands["pre_g"] + bands["pre_n"] + 1e-8)
    ndwi_post = (bands["post_g"] - bands["post_n"]) / (bands["post_g"] + bands["post_n"] + 1e-8)
    
    # 只要灾前或是灾后表现出水体特征，全部判定为水系干扰区 (免疫)
    is_water = (ndwi_pre > CONFIG["NDWI_WATER_THRESH"]) | (ndwi_post > CONFIG["NDWI_WATER_THRESH"])
    
    # 3. 物理约束综合判定
    raw_opt_mask = (delta_ndvi > CONFIG["NDVI_DROP_THRESH"]) & (bands["slope"] > CONFIG["SLOPE_THRESH"]) & (~is_water) & valid_mask
    
    # 4. CV 计算机视觉形态学清理
    clean_opt_mask, labels = cv_clean_mask(raw_opt_mask)
    
    # 5. 保存最终的光学掩膜 (0/1 二值面)
    mask_meta = ref_meta.copy()
    mask_meta.update({"count": 1, "dtype": rasterio.uint8, "nodata": 0})
    with rasterio.open(CONFIG["OUTPUT_MASK"], 'w', **mask_meta) as dst:
        dst.write(clean_opt_mask.astype(np.uint8), 1)

    print(f"      -> 高纯度光学面锁定像素: {np.sum(clean_opt_mask)} 个，已保存为最终 Mask。")
    # ==========================================================
    # 🌟 新增：将二值掩膜直接矢量化为 Shapefile
    # ==========================================================
    print("      -> 正在将光学掩膜矢量化为 Shapefile...")
    geoms = []
    values = []
    for geom, val in shapes(clean_opt_mask.astype(np.uint8), 
                            mask=clean_opt_mask, 
                            transform=ref_meta['transform']):
        geoms.append(shape(geom))
        values.append(int(val))

    if len(geoms) > 0:
        gdf = gpd.GeoDataFrame(
            {"value": values},  # 简单属性字段（1=滑坡）
            geometry=geoms,
            crs=ref_meta['crs']
        )

        gdf.to_file(CONFIG["OUTPUT_MASK_VECTOR"], driver="ESRI Shapefile")
        print(f"      -> ✅ 光学掩膜 Shapefile 已保存: {CONFIG['OUTPUT_MASK_VECTOR']}")
    else:
        print("      -> ⚠️ 未检测到可矢量化区域")
    return clean_opt_mask, labels


def track_and_vectorize_instances(ref_meta, labels):
    """
    模块 3：GIS 追踪与属性赋能引擎 (降水主导 + 时间锁修复版)。
    """
    print("\n[3/3] 启动 GIS 追踪引擎：降水主导溯源，SAR逆向验证与生命周期挂载...")
    
    # 🌟 获取光学影像的绝对时间边界，作为时空锁，防止锁定到灾后的突变 (修复了跑到12月的问题)
    opt_start_date = extract_date_from_filename(CONFIG["S2_PRE_EVENT"])
    opt_end_date = extract_date_from_filename(CONFIG["S2_POST_EVENT"])
    print(f"      [时间锁] 严格限定诱发溯源区间: {opt_start_date.strftime('%Y-%m-%d')} 至 {opt_end_date.strftime('%Y-%m-%d')}")
    
    # 提取所有独立的滑坡实例 ID (跳过背景 0)
    instance_ids = np.unique(labels)[1:] 
    if len(instance_ids) == 0:
        print("      [提示] 研究区内未发现符合阈值的滑坡实例。")
        return

    # --- 加载时序追踪数据源 (SAR & GPM) ---
    sar_files = sorted(glob.glob(os.path.join(CONFIG["SAR_DIR"], "*.tif")))
    sar_dates, sar_stack, sar_diff_stack = [], None, None
    if len(sar_files) >= 2:
        sar_dates = [extract_date_from_filename(f) for f in sar_files]
        sar_stack = np.array([align_raster(f, ref_meta)[0] for f in tqdm(sar_files, desc="      -> 加载 SAR 微波时序栈")])
        sar_diff_stack = np.abs(sar_stack[1:] - sar_stack[:-1]) # 计算相邻时间步形变绝对值
    else:
        print("      [警告] 未发现充足的 SAR 数据，将跳过形变与发生期追踪。")

    gpm_files = sorted(glob.glob(os.path.join(CONFIG["GPM_DIR"], "*.tif")))
    gpm_dates, gpm_stack = [], None
    if gpm_files:
        gpm_dates = [extract_date_from_filename(f) for f in gpm_files]
        gpm_stack = np.array([align_raster(f, ref_meta, resampling_method=Resampling.nearest)[0] for f in tqdm(gpm_files, desc="      -> 加载 GPM 降水时序栈")])
    else:
        print("      [警告] 未发现 GPM 数据，将跳过极端降水溯源。")

    # 用于保存每个多边形矢量属性的列表
    records = []

    # 遍历每一个独立的滑坡斑块
    for idx, inst_id in enumerate(instance_ids):
        inst_mask = (labels == inst_id)
        
        # 初始化当前滑坡的基础属性表
        inst_attr = {
            "Inst_ID": int(inst_id),
            "Area_px": int(np.sum(inst_mask)), # 面积大小(像素)
            "S_Start": "Unknown",              # 崩塌形变起始日
            "S_End": "Unknown",                # 崩塌形变截止日
            "Rain_Max": 0.0,                   # 触发最大面雨量
            "Rain_Date": "Unknown",            # 触发降水精确日
            "Post_Def": 0.0,                   # 灾后残余形变量
            "Is_Active": 0                     # 灾后是否活跃 (0:稳定 1:活跃)
        }
        
        rain_date = None
        
        # ========================================================
        # 🌟 追踪 1: 降水主导 (GPM-First) - 寻找“3天前期累积降雨”的极端峰值
        # ========================================================
        if gpm_stack is not None:
            # 必须在两次光学拍照的【时间锁】之间寻找降雨诱因
            valid_gpm_indices = [i for i, d in enumerate(gpm_dates) if opt_start_date <= d <= opt_end_date]
            if valid_gpm_indices:
                inst_gpm_window = gpm_stack[valid_gpm_indices]
                inst_daily_rain = np.max(inst_gpm_window[:, inst_mask], axis=1) 
                
                # 计算 3 天滑动累积降雨量 (捕捉连续多日暴雨导致的土壤饱和)
                if len(inst_daily_rain) >= 3:
                    cum_rain_3d = np.convolve(inst_daily_rain, np.ones(3, dtype=int), 'valid')
                    peak_cum_idx = np.argmax(cum_rain_3d)
                    peak_idx = peak_cum_idx + 2 # 取 3天暴雨过程的最后一天作为滑坡诱发日
                else:
                    peak_idx = np.argmax(inst_daily_rain)
                    
                rain_date = gpm_dates[valid_gpm_indices[peak_idx]]
                inst_attr["Rain_Date"] = rain_date.strftime('%Y-%m-%d')
                inst_attr["Rain_Max"] = round(float(inst_daily_rain[peak_idx]), 2)
                
        # ========================================================
        # 🌟 追踪 2: 雷达验证 (SAR Bracketing) - 逆向锁定形变突变窗口
        # ========================================================
        s_end_idx = None
        if sar_stack is not None and rain_date is not None:
            inst_sar_diff = np.sum(sar_diff_stack[:, inst_mask], axis=1) / np.sum(inst_mask)
            
            # 寻找降雨日之后的第一景 SAR 影像 (灾后首图 S_End)
            for i, d in enumerate(sar_dates):
                if d >= rain_date:
                    s_end_idx = i
                    break
                    
            if s_end_idx is not None and s_end_idx > 0:
                s_start_idx = s_end_idx - 1 # 降雨日之前的一景 SAR (灾前末图 S_Start)
                t_start, t_end = sar_dates[s_start_idx], sar_dates[s_end_idx]
                inst_attr["S_Start"] = t_start.strftime('%Y-%m-%d')
                inst_attr["S_End"] = t_end.strftime('%Y-%m-%d')
                
                # ========================================================
                # 🌟 追踪 3: 释放 12 月份影像的价值 - 灾后残余形变追踪
                # ========================================================
                # 从灾后首图之后的所有差值，全部累加起来
                post_event_diffs = inst_sar_diff[s_start_idx + 1:]
                if len(post_event_diffs) > 0:
                    inst_attr["Post_Def"] = round(float(np.sum(post_event_diffs)), 4)
                    if np.mean(post_event_diffs) > 0.015:
                        inst_attr["Is_Active"] = 1
                    
        records.append(inst_attr)

    # ========================================================
    # 🌟 将栅格实例转化为几何多边形 (Polygonization)，挂载属性并导出
    # ========================================================
    print("      -> 正在进行矢量化 (Polygonization) 并封测属性表...")
    geoms, geom_records = [], []
    
    # 利用 rasterio.features.shapes 提取掩膜边界，生成多边形坐标
    for geom, val in shapes(labels.astype(np.int16), mask=(labels > 0), transform=ref_meta['transform']):
        inst_id = int(val)
        geoms.append(shape(geom)) # 转换为 shapely geometry
        # 匹配对应 ID 的属性字典
        matched_attr = next((item for item in records if item["Inst_ID"] == inst_id), {})
        geom_records.append(matched_attr)
        
    # 利用 GeoPandas 组合几何与属性表，生成最终图层
    gdf = gpd.GeoDataFrame(geom_records, geometry=geoms, crs=ref_meta['crs'])
    gdf.to_file(CONFIG["OUTPUT_VECTOR"], driver="ESRI Shapefile")
    
    print(f"\n      -> 🎯 [里程碑] 成功将 {len(gdf)} 个带属性的单体滑坡落盘为 Shapefile！")
    print(f"      -> 📂 矢量路径: {CONFIG['OUTPUT_VECTOR']}")
    print("      -> 💡 你现在可以直接将此 Shapefile 和 Delta NDVI 图层拖入 QGIS 中联合展示。")


# =====================================================================
# 4. 主控函数 (Main Pipeline)
# =====================================================================
def main():
    print("\n" + "#"*65)
    print(" 🚀 高纯度光学识别与时空属性矢量追踪引擎 (降雨主导+时间锁修复版)")
    print("#"*65)
    
    # 模块 1: 建立空间对齐基准
    ref_meta, bands = prepare_spatial_data()
    
    # 模块 2: 生成纯净光学掩膜与 Delta NDVI 变化栅格图
    opt_mask, labels = generate_optical_mask(ref_meta, bands)
    
    # 模块 3: 在光学提取的实例上，提取 SAR 形变与 GPM 降水并生成属性矢量图层
    # track_and_vectorize_instances(ref_meta, labels)

if __name__ == "__main__":
    main()