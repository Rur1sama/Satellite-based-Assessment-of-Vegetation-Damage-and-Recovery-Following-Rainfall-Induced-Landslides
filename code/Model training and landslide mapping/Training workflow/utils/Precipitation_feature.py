"""
==============================================================================
Project       : Slide_dect
Module Name   : Precipitation_feature.py
File Path     : code/Model training and landslide mapping/Training workflow/utils/Precipitation_feature.py
File Created  : 2026-05-28 22:08:24
Description   : 降雨特征提取--特征提取模块
Lines         : 105
==============================================================================
"""

from utils.head import * # 导入所有头文件
from utils.__init__ import * # 初始化参数配置、模型参数

def get_rolling_rain2(gpm_base_dir, pre_year, post_year, ref_profile):
    '''
    安全计算降雨参数

    :param gpm_base_dir: GPM降水栅格数据
    :param pre_year: 灾前年份
    :param post_year: 灾后年份
    :param ref_profile: 空间参考
    :return:  rain_lg50  （大于50mm全年累积降水总量）
    \n        rain_1dmax （全年单日最大累计降水）
    \n        rain_7dmax （全年七日最大累计降水）
    \n        rain_1dmean（全年单日平均累计降水）
    \n        rain_7dmean（全年七日平均累计降水）
    '''
    # 1. 收集前后两个年份目录下的所有降雨 tif
    dir_pre, dir_post = os.path.join(gpm_base_dir, pre_year), os.path.join(gpm_base_dir, post_year)
    files = []
    for d in [dir_pre, dir_post]:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*.tif")))
    files = sorted(files)
    # 2. 如果没有找到文件，直接返回 5 张零图
    if not files:
        shape = (ref_profile['height'], ref_profile['width'])
        return [np.zeros(shape, dtype=np.float32)] * 5
    # 3. 读取第一张图，拿到 GPM 原始网格信息
    with rasterio.open(files[0]) as src0:
        src_crs = src0.crs
        src_transform = src0.transform
        src_h, src_w = src0.height, src0.width
    # 4. 在原始网格上直接累计统计，避免先生成超大的 stack
    rain_lg50 = np.zeros((src_h, src_w), dtype=np.float32)      # 累计 >=50 的降雨量
    rain_1dmax = np.full((src_h, src_w), np.nan, dtype=np.float32)  # 单日最大值
    rain_1dsum = np.zeros((src_h, src_w), dtype=np.float32)      # 单日总和
    rain_7dmax = np.full((src_h, src_w), np.nan, dtype=np.float32)  # 7日滑动最大值
    rain_7dsum = np.zeros((src_h, src_w), dtype=np.float32)      # 所有 7 日窗口总和
    # 5. 维护一个 7 天滑动窗口，只保存最近 7 张图
    window = []
    win_sum = np.zeros((src_h, src_w), dtype=np.float32)
    # 6. 逐日读取、逐日更新，不再一次性堆栈
    for f in tqdm(files, desc="统计降雨指标", unit="day",leave=False):
        with rasterio.open(f) as src:
            arr = src.read(1).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
        # 6.1 把 NaN 临时当 0 参与求和
        arr0 = np.nan_to_num(arr, nan=0.0)
        # 6.2 累计 >=50 的降雨量
        rain_lg50 += np.where(arr >= 50, arr0, 0.0)
        # 6.3 单日最大值、单日总和
        rain_1dmax = np.fmax(rain_1dmax, arr)
        rain_1dsum += arr0
        # 6.4 更新 7 日滑动窗口
        win_sum += arr0
        window.append(arr0)
        if len(window) > 7:
            win_sum -= window.pop(0)
        # 6.5 满 7 天后，更新 7 日最大值和 7 日窗口总和
        if len(window) == 7:
            rain_7dmax = np.fmax(rain_7dmax, win_sum)
            rain_7dsum += win_sum
    # 7. 计算平均值
    rain_1dmean = rain_1dsum / max(len(files), 1)
    if len(files) >= 7:
        rain_7dmean = rain_7dsum / (len(files) - 6)
    else:
        rain_7dmean = rain_1dmean.copy()
    # 8. 如果最后结果和参考网格不一致，再统一重采样到 ref_profile
    def _resample_to_ref(arr):
        if (arr.shape == (ref_profile['height'], ref_profile['width']) and
            src_crs == ref_profile['crs'] and
            src_transform == ref_profile['transform']):
            return arr.astype(np.float32)

        dst = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)
        reproject(
            arr, dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'],
            resampling=Resampling.bilinear
        )
        return dst

    # 9. 返回 5 张统计图，顺序保持不变
    return (
        _resample_to_ref(rain_lg50),
        _resample_to_ref(rain_1dmax),
        _resample_to_ref(rain_7dmax),
        _resample_to_ref(rain_1dmean),
        _resample_to_ref(rain_7dmean)
    )