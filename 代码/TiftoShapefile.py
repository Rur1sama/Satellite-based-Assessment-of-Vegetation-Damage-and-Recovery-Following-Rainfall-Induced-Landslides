import os
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

def tif_2_shp(tif_path, shp_path, threshold=0.8, min_pixels=4):
    '''
    将滑坡概率 TIFF 转换为 Shapefile 矢量文件。

    :param tif_path: 输入的 TIFF 概率文件路径
    :param shp_path: 输出的 Shapefile 保存路径
    :param threshold: 判定为滑坡的概率阈值 (0.0 - 1.0)
    :param min_pixels: 过滤孤立微小斑块的最小像元数阈值
    :return: None
    '''
    if not os.path.exists(tif_path):
        print(f"错误: 找不到输入的 TIFF 文件: {tif_path}")
        return

    print(f"[*] 正在读取栅格数据: {os.path.basename(tif_path)}")
    with rasterio.open(tif_path) as src:
        profile = src.profile
        crs = src.crs
        transform = src.transform
        res = abs(transform[0])  # 像元分辨率
        data = src.read(1)
        
    print(f"[*] 正在提取概率 >= {threshold} 的滑坡区域...")
    # 二值化，过滤掉 NaN 值
    valid_mask = np.isfinite(data)
    binary_map = ((data >= threshold) & valid_mask).astype(np.uint8)
    
    # 利用 rasterio 提取边界
    shape_generator = shapes(binary_map, mask=(binary_map == 1), transform=transform)
    
    records = []
    for geom, val in shape_generator:
        poly = shape(geom)
        # 面积过滤 (poly.area / 像元面积 = 该斑块包含的像元数)
        if poly.area / (res * res) >= min_pixels:
            records.append({
                "geometry": poly,
                "Thresh": float(threshold),
                "Area_m2": round(poly.area, 2)
            })
            
    if not records:
        print("[-] 未能提取到任何满足条件的滑坡几何体。")
        return
        
    print(f"[*] 正在生成矢量要素，共计 {len(records)} 个原始斑块...")
    gdf = gpd.GeoDataFrame(records, crs=crs)
    
    # 融合相邻的边界（使滑坡面整体性更强）
    gdf['gid'] = 1
    gdf = gdf.dissolve(by='gid').reset_index(drop=True)
    gdf["Area_m2"] = np.round(gdf.geometry.area, 2)
    
    # 保存文件
    print(f"[*] 正在写入 Shapefile...")
    gdf.to_file(shp_path, encoding='utf-8')
    print(f"[✓] 转换完成！Shapefile 已成功保存至:\n    {shp_path}")

if __name__ == "__main__":
    # 配置你的路径
    INPUT_TIF = r"E:\python_base\Slide_dect\应用\资兴市\Output\Inference_Results\Final_Ensemble_Landslide_23-24.tif"
    OUTPUT_SHP = r"E:\python_base\Slide_dect\应用\资兴市\Output\Inference_Results\Final_Ensemble_Landslide_23-24.shp"
    
    tif_2_shp(INPUT_TIF, OUTPUT_SHP, threshold=0.8, min_pixels=4)
