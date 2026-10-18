"""
==============================================================================
Project       : Slide_dect
Module Name   : label_process.py
File Path     : code/Model training and landslide mapping/Training workflow/utils/label_process.py
File Created  : 2026-05-28 22:11:48
Description   : 训练集构建、随机抽样拆分和空间分组
Lines         : 120
==============================================================================
"""

from utils.head import * # 导入所有头文件
from utils.__init__ import * # 初始化参数配置、模型参数
from utils.base_utils import *
from utils.Optical_feature import build_optical_features
from utils.Terrain_feature import build_terrain_features
from utils.Precipitation_feature import get_rolling_rain2

def remap_label(x):
    '''
    将原始人工标注编码归并为背景、滑坡和其他扰动三类。

    :param x: 原始标签编码
    :return: 三分类标签，其中 0 为背景、1 为滑坡、2 为其他扰动
    '''
    if x in [1, 2, 4]: return 1
    elif x in [5, 6]: return 2
    else: return 0 # 3、7、8为背景

def build_feature_dataframe(year_pair):
    '''
    构建当前年份对的像素级特征 DataFrame，即训练集。

    :param year_pair: 年份对
    :return: df      未抽样训练集
    \n       profile 矩阵模板
    \n       h       模板高
    \n       w       模板宽
    '''
    pre_year, post_year, pre_path, post_path, label_path = get_year_input_paths(year_pair)

    with rasterio.open(post_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)

        with tqdm(total=4, desc=f"  [特征构建] 正在初始化管道...", position=2, leave=False) as pbar:
            pbar.set_description(f"  [特征构建] 1/4 计算光学指数 (NDVI / Delta-NDVI / NDWI)")
            ndvi_post, delta_ndvi, ndwi_post = build_optical_features(src, pre_path, profile)
            pbar.update(1)

            pbar.set_description(f"  [特征构建] 2/4 重采样并生成地形因子 (坡度/坡向/曲率)，并计算 TWI 地形水文指数")
            dem, res, curvature, aspect, aspect_ns,aspect_ew, slope,twi = build_terrain_features(profile)
            pbar.update(1)

            pbar.set_description(f"  [特征构建] 3/4 载入并分析多时序降雨数据 (进入GPM提取)")
            r_lg50, r_1dmax, r_7dmax, r_1dmean, r_7dmean = get_rolling_rain2(
                CONFIG["GPM_DIR"], pre_year, post_year, profile
            )
            pbar.update(1)

            pbar.set_description(f"  [特征构建] 4/4 矢量标签栅格化 & 全区 DataFrame 展开清洗")
            h, w = profile['height'], profile['width']
            rows, cols = np.indices((h, w))
            block = (rows // 50) * 1000 + (cols // 50)
            labels = rasterize_year_labels(label_path, profile, (h, w))

            df = pd.DataFrame({
                "idx": np.arange(h * w),
                "delta_ndvi": delta_ndvi.flatten(),
                "ndvi": ndvi_post.flatten(),
                "ndwi": ndwi_post.flatten(),
                "elevation": dem.flatten(),
                "curvature": curvature.flatten(),
                "aspect": aspect.flatten(),
                "aspect_ns": aspect_ns.flatten(),
                "aspect_ew": aspect_ew.flatten(),
                "slope": slope.flatten(),
                "rain_lg50": r_lg50.flatten(),
                "rain_1dmax": r_1dmax.flatten(),
                "rain_7dmean": r_7dmean.flatten(),
                "rain_7dmax": r_7dmax.flatten(),
                "twi": twi.flatten(),
                "block": block.flatten(), # 空间分组
                "label": labels.flatten()
            })
            df['feat_rain_slope'] = df['rain_7dmax'] * np.tan(np.radians(df['slope']))
            df['feat_veg_fail'] = df['delta_ndvi'] * np.tan(np.radians(df['slope']))
            df = df.dropna().reset_index(drop=True)
            df["label"] = df["label"].apply(remap_label)
            pbar.update(1)

    return df, profile, h, w


def sample_training_dataframe(df, year_pair, train_data_dir):
    '''
    为指定年份对创建或读取共享训练样本。
    四个分类器复用同一份类别抽样结果和空间交叉验证折编号。

    :param df: 未抽样的全图像元特征表
    :param year_pair: Sentinel-2 年份对
    :param train_data_dir: 共享训练样本保存目录
    :return: 包含标签、空间块和固定折编号的训练样本表
    '''

    os.makedirs(train_data_dir, exist_ok=True)
    train_path = os.path.join(
        train_data_dir,
        f"train_{year_pair}.csv"
    )

    required_cols = {"idx", "label", "block", "fold_id"}

    # 已存在时直接复用，防止不同算法重新抽样
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)

        missing = required_cols.difference(df_train.columns)
        if missing:
            raise RuntimeError(
                f"{train_path} 缺少字段 {sorted(missing)}，"
                "请删除该CSV后重新生成。"
            )

        log(f"复用共享训练样本: {train_path}")
        return df_train

    seed = CONFIG["RANDOM_STATE"]

    df_ls = df[df.label == 1].copy()
    df_dist_pool = df[df.label == 2]
    df_bg_pool = df[df.label == 0]

    df_dist = df_dist_pool.sample(
        n=min(len(df_ls) * 8, len(df_dist_pool)),
        random_state=seed
    )
    df_bg = df_bg_pool.sample(
        n=min(len(df_ls) * 10, len(df_bg_pool)),
        random_state=seed + 1
    )

    df_train = (
        pd.concat([df_ls, df_dist, df_bg], ignore_index=True)
        .sample(frac=1, random_state=seed + 2)
        .reset_index(drop=True)
    )

    # 固定每个空间块所属的验证折
    fold_id = np.full(len(df_train), -1, dtype=np.int16)
    gkf = GroupKFold(n_splits=CONFIG["N_SPLITS"])
    dummy_x = np.zeros((len(df_train), 1), dtype=np.uint8)

    splits = gkf.split(
        dummy_x,
        df_train["label"],
        groups=df_train["block"]
    )
    for current_fold, (_, te) in enumerate(splits, start=1):
        fold_id[te] = current_fold

    if np.any(fold_id < 1):
        raise RuntimeError("部分样本没有获得fold编号。")

    df_train["fold_id"] = fold_id

    # 同一个空间块只能属于一个fold
    max_fold_count = (
        df_train.groupby("block")["fold_id"]
        .nunique()
        .max()
    )
    if max_fold_count != 1:
        raise RuntimeError("同一空间块被分配到了多个fold。")

    df_train.to_csv(train_path, index=False)
    log(f"已生成共享训练样本: {train_path}")

    return df_train

def split_training_inputs(df_train):
    '''
    从共享训练样本表中拆分模型输入、标签、空间块和固定折编号。

    :param df_train: 包含固定折编号的共享训练样本表
    :return: 模型特征、类别标签、空间块编号和交叉验证折编号
    '''
    X = df_train.drop(
        columns=MODEL_FEATURE_DROP_COLUMNS,
        errors="ignore"
    )
    y = df_train["label"].astype(int)
    groups = df_train["block"]
    fold_ids = df_train["fold_id"].astype(int)

    return X, y, groups, fold_ids
