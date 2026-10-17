"""
==============================================================================
Project       : Slide_dect
Module Name   : 加权结果.py
File Path     : 代码/机器学习训练/加权结果.py
File Created  : 2026-05-14 10:58:45
Author        : leeyukhang
Description   : 加权模型计算结果
==============================================================================
"""
#!/usr/bin/env python
# coding=utf-8
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer

# =========================================================
# 1. 集成配置
# =========================================================
# 定义各模型预测结果的根目录
BASE_DIR = r"E:\python_base\Slide_dect\Output\ML_Results"
CONFIG = {
    "MODELS": {
        "LightGBM": os.path.join(BASE_DIR, "lightBGM"),
        "Random Forest": os.path.join(BASE_DIR, "Random Forest"),
        "XGBoost": os.path.join(BASE_DIR, "XGBoost"),
        "CatBoost": os.path.join(BASE_DIR, "CatBoost")
    },
    # 权重配置：建议根据各模型的验证集性能分配 (总和应为1.0)
    # 示例：CatBoost和LGBM通常在结构化数据表现更好，给予更高权重
    "WEIGHTS": {
        "LightGBM": 0.25,
        "XGBoost": 0.25,
        "CatBoost": 0.25,
        "Random Forest": 0.25
    },
    "OUTPUT_DIR": os.path.join(BASE_DIR, "Ensemble_Final"),
    # "YEAR_PAIRS": ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"]
    "YEAR_PAIRS": ["23-24"]
}
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

def run_ensemble(year_pair):
    print(f"正在处理年份: {year_pair} ...")
    ensemble_sum = None
    valid_mask = None
    meta = None
    # 2. 迭代载入各模型结果
    for model_name, model_path in CONFIG["MODELS"].items():
        tif_path = os.path.join(model_path, f"Landslide_{year_pair}.tif")
        if not os.path.exists(tif_path):
            print(f"  [Error] 跳过 {model_name}: 找不到文件 {tif_path}")
            continue
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)
            # 记录元数据（以第一个成功读取的模型为准）
            if meta is None:
                meta = src.meta.copy()
                # 确保输出格式为 float32
                meta.update(dtype=rasterio.float32, nodata=np.nan)
                ensemble_sum = np.zeros_like(data)
                valid_mask = np.ones_like(data, dtype=bool)
                print(f"meta の crs:{meta['crs']}")
            # 获取当前权重
            w = CONFIG["WEIGHTS"].get(model_name,0)
            # 处理无效值（通常 < 0 为 NoData）
            # 只有当所有模型在该点都有值时，才参与集成
            invalid = (data < 0) | (np.isnan(data))
            data[invalid] = 0  # 临时设为0进行加权和
            valid_mask &= (~invalid)
            
            ensemble_sum += data * w
            print(f"  + 已混入 {model_name} (Weight: {w})")

    # 3. 生成并保存最终结果
    if ensemble_sum is not None:
        # 将无信号区域（如全图掩膜外）还原为 NaN
        ensemble_sum[~valid_mask] = np.nan
        out_path = os.path.join(CONFIG["OUTPUT_DIR"], f"Ensemble_Final_{year_pair}.tif")
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(ensemble_sum, 1)
        # =========================================================
        # 4. 可视化结果
        # =========================================================
        fig, ax = plt.subplots(figsize=(16, 12))
         # 设置字体为 SimHei（黑体）并解决负号显示问题
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        # =========================================================
        # UTM -> 经纬度转换
        # =========================================================
        transform = meta['transform']
        height, width = ensemble_sum.shape
        # 四角坐标（UTM）
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        # 转换到WGS84
        transformer = Transformer.from_crs(
            meta['crs'],
            "EPSG:4326",
            always_xy=True
        )
        lon_min, lat_max = transformer.transform(left, top)
        lon_max, lat_min = transformer.transform(right, bottom)
        # 绘图
        img = ax.imshow(
            ensemble_sum,
            cmap='terrain', # terrain 、viridis等色系任君选择
            vmin=0,
            vmax=1,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin='upper'
        )
        # =========================================================
        # 地理坐标轴
        # =========================================================
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)

        # 经纬网
        ax.grid(
            True,
            linestyle='--',
            linewidth=0.8,
            color='gray',
            alpha=0.7
        )
        # 自动刻度
        ax.tick_params(
            axis='both',
            labelsize=10
        )

        # 色标
        cbar = plt.colorbar(img, ax=ax, shrink=0.75)
        cbar.set_label('预测概率', fontsize=12)
        # 标题
        ax.set_title(
            f"合成后滑坡预测概率图 - {year_pair}",
            fontsize=18,
            fontweight='bold'
        )
        # =========================================================
        # 添加比例尺
        # =========================================================
        
        pixel_size = abs(meta['transform'][0]) # 获取像元大小（单位通常为米）
        # 1 km对应经度跨度（近似）
        meters_per_degree = 111320 * np.cos(np.deg2rad(lat_min))
        segment_km = 1
        segment_m = segment_km * 1000
        segment_lon = segment_m / meters_per_degree
        # 比例尺位置（经纬度）
        bar_lon0 = lon_min + (lon_max - lon_min) * 0.04
        bar_lat0 = lat_min + (lat_max - lat_min) * 0.035
        bar_height = (lat_max - lat_min) * 0.008
        # 黑白交替
        for i in range(5):
            color = 'black' if i % 2 == 0 else 'white'
            rect = plt.Rectangle(
                (bar_lon0 + i * segment_lon, bar_lat0),
                segment_lon,
                bar_height,
                facecolor=color,
                edgecolor='black',
                linewidth=1.2
            )
            ax.add_patch(rect)
        # 刻度
        for i in range(6):
            x = bar_lon0 + i * segment_lon
            ax.text(
                x,
                bar_lat0 - bar_height * 1.8,
                f"{i}",
                ha='center',
                va='top',
                fontsize=10
            )
        # km标签
        ax.text(
            bar_lon0 + 5 * segment_lon + segment_lon * 0.3,
            bar_lat0 - bar_height * 1.8,
            "km",
            fontsize=11
        )
        # =========================================================
        # 保存
        # =========================================================
        plt.savefig(
            os.path.join(CONFIG["OUTPUT_DIR"], f"Preview_{year_pair}.png"),
            dpi=600,
            bbox_inches='tight'
        )
        plt.close()
        print(f"  [Success] 集成结果保存至: {out_path}")
    else:
        print(f"  [Failed] {year_pair} 无可用模型结果进行集成")
# =========================================================
# 入口
# =========================================================
if __name__ == "__main__":
    print("="*50)
    print("开始多模型加权集成反演...")
    print("="*50)
    for yp in CONFIG["YEAR_PAIRS"]:
        run_ensemble(yp)
    print("\n任务全部完成！请在 Ensemble_Final 文件夹查看结果。")