"""
==============================================================================
Project       : Slide_dect
Module Name   : 训练2.py
File Path     : 代码/机器学习训练/训练2.py
File Created  : 2026-05-07 19:46:00
Author        : leeyukhang
Description   : Random Forest 训练
==============================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier  # ⭐ 替换为随机森林
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import datetime
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns
from scipy.stats import pearsonr
import joblib 
import math 
import matplotlib.image as mpimg 
# 计算TWI
import tempfile
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
# 屏蔽预料内的数学计算警告 (NDVI计算中的空值或除零)
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# =========================================================
# 配置 (已修改输出目录为 RF)
# =========================================================
CONFIG = {
    "OPTICAL_DIR": r"E:\python_base\Slide_dect\Input\光学栅格",
    "LABEL_ROOT": r"E:\python_base\Slide_dect\Input\样本面手动标注结果",
    "DEM_PATH": r"E:\python_base\Slide_dect\Input\高程数据\ALOS_PALSAR_DEM.tif",
    "GPM_DIR": r"E:\python_base\Slide_dect\Input\GPM_Time_Series",
    "OUTPUT_DIR": r"E:\python_base\Slide_dect\Output\ML_Results\RF",  # ⭐ 适配训练2的输出目录
    "BANDS": {"RED": 4, "NIR": 8}
}

# 子目录定义
ROC_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "ROC_curves")
PR_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "PR_curves")
CM_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "CM_folds")
CORR_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "Correlation_plots")
MODEL_SAVE_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "Saved_Models")
TRAIN_DATA_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "Training_Data_CSVs")
FI_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "Feature_Importance")
Global_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "global_summary")

for d in [ROC_DIR, PR_DIR, CM_DIR, CORR_DIR, MODEL_SAVE_DIR, TRAIN_DATA_DIR, FI_DIR, Global_DIR]:
    os.makedirs(d, exist_ok=True)

''' 
=========================================================
 工具函数部分 (完全沿用训练1)
=========================================================
'''

def log(msg): 
    '''
    记录运行日志并打印输出

    :param msg: 需要记录的字符串消息
    :return: None
    '''
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(os.path.join(CONFIG["OUTPUT_DIR"], "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return None

def smart_align(src_path, ref_profile, band_idx=1):
    '''
    将栅格影像根据参考影像重投影并对齐

    :param src_path: 待处理栅格的路径
    :param ref_profile: 参考影像的配置参数（profile）
    :param band_idx: 需要读取的波段索引
    :return: 对齐后的二维numpy数组
    '''
    with rasterio.open(src_path) as src:
        dest = np.empty((ref_profile['height'], ref_profile['width']), dtype=np.float32)
        reproject(rasterio.band(src, band_idx), dest, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_profile['transform'], dst_crs=ref_profile['crs'], resampling=Resampling.bilinear)
        if src.nodata is not None:
            dest[dest == np.float32(src.nodata)] = np.nan 
        return dest

def safe_ndvi(nir, red):
    '''
    安全计算NDVI，防止除以零或无效值

    :param nir: 近红外波段数组
    :param red: 红光波段数组
    :return: NDVI结果数组
    '''
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndvi[(nir + red) <= 0] = np.nan
    return ndvi

def get_rolling_rain2(gpm_base_dir, pre_year, post_year, ref_profile):
    '''
    提取降雨特征：包含暴雨天数、日/周极值及均值特征

    :param gpm_dir: GPM降水数据所在目录
    :param ref_profile: 空间参考配置
    :param pre_year:
    :param post_year: 
    :return:  rain_lg50  （大于50mm全年累积降水总量）
    \n        rain_1dmax （全年单日最大累计降水）
    \n        rain_7dmax （全年七日最大累计降水）
    \n        rain_1dmean（全年单日平均累计降水）
    \n        rain_7dmean（全年七日平均累计降水）
    '''
    dir_pre = os.path.join(gpm_base_dir, pre_year)
    dir_post = os.path.join(gpm_base_dir, post_year)
    
    files = []
    for d in [dir_pre, dir_post]:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*.tif")))
        else:
            log(f"警告: 找不到降雨数据文件夹: {d}")
            
    files = sorted(files)
    
    if not files:
        log(f"警告: {pre_year} 和 {post_year} 的降雨文件夹中均未找到 .tif 文件")
        shape = (ref_profile['height'], ref_profile['width'])
        return [np.zeros(shape)] * 5
        
    stack = np.array([smart_align(f, ref_profile) for f in files])
    rolling = np.zeros_like(stack)
    
    for i in range(6, len(stack)):
        rolling[i] = np.nansum(stack[i-6:i+1], axis=0)
        
    return np.nansum(np.where(stack >= 50, stack, 0), axis=0), \
           np.nanmax(stack, axis=0), \
           np.nanmax(rolling, axis=0), \
           np.nanmean(stack, axis=0), \
           np.nanmean(rolling, axis=0)

def remap_label(x):
    '''
    将原始地物类别映射为如下分类：背景(0)、滑坡(1)、扰动(2)

    :param x: 原始标签值
    :return: 映射后的整数标签
    '''
    if x in [1, 2, 4]: return 1
    elif x in [5, 6]: return 2
    else: return 0

def plot_feature_correlation(df_features, year_pair, out_dir):# 计算并绘制带显著性星号特征相关性热力图
    '''
    计算并绘制带显著性星号特征相关性热力图

    :param df_features: 特征DataFrame
    :param year_pair: 年份标识 (图片名称)
    :return: None
    '''
    corr = df_features.corr()
    annot_matrix = corr.copy().astype(str)
    for i, col_i in enumerate(df_features.columns):
        for j, col_j in enumerate(df_features.columns):
            if i != j:
                r, p = pearsonr(df_features[col_i], df_features[col_j])
                star = None
                if p <0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p <0.05:
                    star = "*"
                else:
                    star = ""
                annot_matrix.iloc[i, j] = star if i < j else f"{r:.2f}"
            else:
                annot_matrix.iloc[i, j] = "1.00"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=annot_matrix, fmt="", cmap='RdBu', center=0, square=True)
    plt.title(f"Feature Correlation - {year_pair}")
    plt.savefig(os.path.join(out_dir, f"Corr_Star_{year_pair}.png"), dpi=600, bbox_inches='tight')
    plt.close()

def create_summary_grid(image_dir, prefix, out_filename, cols=3):
    '''
    搜索指定目录下符合前缀的图片，拼接并保存为一个大图网格

    :param image_dir: 图片所在目录
    :param prefix: 文件名前缀
    :param out_filename: 输出文件名
    :param cols: 网格列数
    :return: None
    '''
    img_files = sorted(glob.glob(os.path.join(image_dir, f"{prefix}*.png")))
    if not img_files: return
    n = len(img_files)
    rows = math.ceil(n / cols)
    tmp_img = mpimg.imread(img_files[0])
    h, w = tmp_img.shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6 * (h/w)))
    axes = np.array([axes]).flatten()
    for i, ax in enumerate(axes):
        if i < n: ax.imshow(mpimg.imread(img_files[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, out_filename), dpi=600, bbox_inches='tight')
    plt.close()

def calculate_curvature(dem, res):
    """
    计算曲率：高程的二阶导数
    """
    # 计算一阶导数（梯度）
    dy, dx = np.gradient(dem, res)
    # 计算二阶导数
    d2y, _ = np.gradient(dy, res)
    _, d2x = np.gradient(dx, res)
    return d2x + d2y

def calculate_twi_pysheds(dem, res, ref_profile, dirmap=(64, 128, 1, 2, 4, 8, 16, 32)):
    '''
    基于 pysheds 官方流程计算 TWI。 

    :param dem: 2D ndarray.已经通过 smart_align 对齐到 post 影像网格的 DEM。
    :param res: float. 像元分辨率。
    :param ref_profile: dict. 与 dem 对应的 raster profile（用于写临时 GeoTIFF）。
    :param dirmap: tuple, defaults to (64, 128, 1, 2, 4, 8, 16, 32) D8 流向编码。
    :return: 2D ndarray; twi
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
        tmp_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=nodata
        )

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

def calculate_global_metrics(all_true, all_prob, all_pred, all_train_dfs):
    '''
    汇总所有年份数据，计算并绘制全局评估指标（CM百分比图, ROC, PR，以及全局特征相关性）

    :param all_true: 真实标签列表
    :param all_prob: 预测滑坡概率列表
    :param all_pred: 预测分类结果列表
    :return: None
    '''
    all_true, all_prob, all_pred = np.array(all_true), np.array(all_prob), np.array(all_pred)
    all_true_bin = (all_true == 1).astype(int)
    all_pred_bin = (all_pred == 1).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(all_true_bin, all_pred_bin).ravel()
    total = len(all_true_bin)
    
    log("\n【全局汇总结果】")
    log(f"正确的  (TP%) : {tp/total:.2%}\n错判了 (FP%) : {fp/total:.2%}\n漏判了 (FN%) : {fn/total:.2%}")
    log(f"正确背景(TN%) : {tn/total:.2%}\n正确率 (Acc) : {(tp+tn)/total:.2%}")

    # 全局9格百分比混淆矩阵图
    cm_perc = confusion_matrix(all_true, all_pred).astype('float') / len(all_true)
    plt.figure(figsize=(8, 7)); sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues',  
                xticklabels=['BG', 'Slide', 'Disturb'],
                yticklabels=['BG', 'Slide', 'Disturb'])
    plt.savefig(os.path.join(Global_DIR, "Global_CM_9DPercentage.png"), dpi=600); plt.close()
    
    # 全局四格百分比混淆矩阵图
    cm_4dim_global = np.array([[tp/total, fp/total], [fn/total, tn/total]])
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_4dim_global, annot=True, fmt='.2%', cmap='Blues', annot_kws={"size": 16},
                xticklabels=['TP', 'FP'],
                yticklabels=['FN', 'TN'])
    # plt.title("Global Confusion Matrix - 4 Dimensions (%)")
    plt.savefig(os.path.join(Global_DIR, "Global_CM_4Dimensions.png"), dpi=600); plt.close()

    # 全局 ROC 曲线
    fpr, tpr, _ = roc_curve(all_true_bin, all_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6));plt.plot(fpr, tpr, lw=2, label=f'Global ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('Global ROC Curve (Landslide vs Rest)');plt.legend()
    plt.savefig(os.path.join(Global_DIR, "Global_ROC.png"), dpi=600, bbox_inches='tight');plt.close()
    
    # 全局 PR 曲线
    precision_vals, recall_vals, _ = precision_recall_curve(all_true_bin, all_prob)
    plt.figure(figsize=(8, 6));plt.plot(recall_vals, precision_vals, lw=2, label='Global PR')
    plt.xlabel('Recall');plt.ylabel('Precision')
    plt.title('Global Precision-Recall Curve (Landslide vs Rest)');plt.legend()
    plt.savefig(os.path.join(Global_DIR, "Global_PR.png"), dpi=600, bbox_inches='tight');plt.close()
    
    # 全局特征相关性图
    global_train_df = pd.concat(all_train_dfs, ignore_index=True)
    features_to_plot = global_train_df.drop(columns=['label', 'idx', 'block','curvature','aspect'], errors='ignore')
    # 排除非特征列（label, idx, block）后调用已有的绘图函数
    plot_feature_correlation(features_to_plot, "Global_Average", Global_DIR)

# =========================================================
# 核心业务逻辑
# =========================================================

def run_model(year_pair):
    '''
    执行年度模型训练流程：特征提取、交叉验证、模型保存及分折指标绘图

    :param year_pair: 年份字符串 (如 '23-24')
    :return: (year_y_true, year_y_prob, year_y_pred)
    '''
    year_y_true, year_y_prob, year_y_pred = [], [], []
    
    # fi_list = [] # RF用不上fi_list
    
    pre_year, post_year = "20"+year_pair.split("-")[0], "20"+year_pair.split("-")[1]
    pre_path = glob.glob(os.path.join(CONFIG["OPTICAL_DIR"], f"SR2_{pre_year}*", "*.tif"))[0]
    post_path = glob.glob(os.path.join(CONFIG["OPTICAL_DIR"], f"SR2_{post_year}*", "*.tif"))[0]
    label_path = glob.glob(os.path.join(CONFIG["LABEL_ROOT"], f"面{year_pair}", "*.shp"))[0]

    with rasterio.open(post_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)
        red_post, nir_post = src.read(CONFIG["BANDS"]["RED"]).astype(np.float32), src.read(CONFIG["BANDS"]["NIR"]).astype(np.float32)
        ndvi_post = safe_ndvi(nir_post, red_post)
        delta_ndvi = safe_ndvi(smart_align(pre_path, profile, CONFIG["BANDS"]["NIR"]), smart_align(pre_path, profile, CONFIG["BANDS"]["RED"])) - ndvi_post
        dem = smart_align(CONFIG["DEM_PATH"], profile)
        res = abs(profile['transform'][0]) # 获取分辨率
        twi_array = calculate_twi_pysheds(dem, res, profile)

        dy, dx = np.gradient(dem, res)
        curvature = calculate_curvature(dem, res)
        aspect = ((np.degrees(np.arctan2(-dx, dy)) + 360.0) % 360.0) # [-180°, 180°] -> [0°, 360°]
        aspect_rad = np.radians(aspect)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # 传入 CONFIG["GPM_DIR"] 以及解析出来的两个目标年份
        r_lg50, r_1dmax, r_7dmax, r_1dmean, r_7dmean = get_rolling_rain2(CONFIG["GPM_DIR"], pre_year, post_year, profile) # 1dmean不用了
        
        h, w = profile['height'], profile['width']
        rows, cols = np.indices((h, w))
        block = (rows // 50) * 1000 + (cols // 50)
        
        gdf = gpd.read_file(label_path).to_crs(profile['crs'])
        labels = rasterize([(g, int(v)) for g, v in zip(gdf.geometry, gdf['LsBg']) if g and not pd.isna(v)],
                           out_shape=(h, w), transform=profile['transform'], fill=0, dtype=np.int16)
        
        df = pd.DataFrame({
            "idx": np.arange(h*w), 
            "delta_ndvi": delta_ndvi.flatten(), "ndvi": ndvi_post.flatten(),
            "elevation": dem.flatten(), "curvature": curvature.flatten(), 
            "aspect": aspect.flatten(),
            "aspect_ns": np.cos(aspect_rad).flatten(), # 南北向：1为北，-1为南
            "aspect_ew": np.sin(aspect_rad).flatten(), # 东西向：1为东，-1为西
            "slope": slope.flatten(),
            "rain_lg50": r_lg50.flatten(), "rain_1dmax": r_1dmax.flatten(),"rain_7dmean": r_7dmean.flatten(),"rain_7dmax": r_7dmax.flatten(),
            "twi": twi_array.flatten(), # 依然叫 twi
            "block": block.flatten(), "label": labels.flatten()
        })

        # --- [核心插入点] 计算交叉特征 ---
        # 降雨 × 坡度：暴雨 + 陡坡 (注意原特征 slope 是角度，需用 np.radians 转弧度后计算 tan)
        df['feat_rain_slope'] = df['rain_7dmax'] * np.tan(np.radians(df['slope']))
        # 降雨 × 曲率：仅正曲率（汇流地形）积水
        # df['feat_converge'] = df['rain_7dmax'] * (1 + np.maximum(0, df['curvature']))
        # 坡度 × NDVI变化：陡坡区域植被突变
        df['feat_veg_fail'] = df['delta_ndvi'] * np.tan(np.radians(df['slope']))
        # --- [核心插入点] 计算交叉特征 ---
        df = df.dropna().reset_index(drop=True) # 清洗无效值
        df["label"] = df["label"].apply(remap_label) # 重映射标签
    # 绘制特征相关性
    plot_feature_correlation(df.drop(columns=['idx','block','label','curvature','aspect']), year_pair,CORR_DIR) # 记得剔除不要的特征

    # 抽样与数据保存
    df_ls = df[df.label==1] # 滑坡标签为1的抽出
    df_train = pd.concat([
        df_ls,
        df[df.label==2].sample(min(len(df_ls)*8 , len(df[df.label==2])), random_state=42),
        df[df.label==0].sample(min(len(df_ls)*10, len(df[df.label==0])), random_state=42)]).sample(frac=1, random_state=42)
    df_train.to_csv(os.path.join(TRAIN_DATA_DIR, f"train_{year_pair}.csv"), index=False)
    
    # 在这里把不要的特征drop掉
    X, y, groups = df_train.drop(columns=['label','idx','block','curvature','aspect']), df_train['label'], df_train['block']
    models = []
    gkf = GroupKFold(10)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        # ⭐ 模型替换为训练2中的 RandomForestClassifier，保持其参数配置
        model = RandomForestClassifier(
            n_estimators=300,
            class_weight={0:1, 1:12, 2:4},
            n_jobs=-1,
            random_state=42,
            # predict_disable_shape_check=True
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        models.append(model)
        joblib.dump(model, os.path.join(MODEL_SAVE_DIR, f"RF_{year_pair}_Fold{i+1}.pkl"))
        # fi_list.append(model.feature_importances_) # 注释 Feature Importance 相关的逻辑
        
        # 记录验证集结果，return出来
        preds = model.predict(X.iloc[te])
        year_y_true.extend(y.iloc[te].tolist());  
        year_y_pred.extend(preds.tolist())
        year_y_prob.extend(model.predict_proba(X.iloc[te])[:, 1].tolist())


        # 九格百分比混淆矩阵 (三分类视角: 滑坡 vs disturb、BG)
        cm_multi = confusion_matrix(y.iloc[te], preds, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(cm_multi, display_labels=["BG", "Slide", "Disturb"])
        disp.plot(cmap='Blues'); plt.title(f"Fold {i+1}"); 
        plt.savefig(os.path.join(CM_DIR, f"CM_{year_pair}_F{i+1}.png")); plt.close()
        
        # 四格百分比混淆矩阵 (二分类视角: 滑坡 vs 其他)
        y_true_bin = (y.iloc[te] == 1).astype(int)
        y_pred_bin = (preds == 1).astype(int)
        tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        total_f = len(y_true_bin)
        cm_4dim = np.array([[tp_f/total_f, fp_f/total_f], [fn_f/total_f, tn_f/total_f]])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_4dim, annot=True, fmt='.2%', cmap='Greens', cbar=False,
                    xticklabels=['Pred Slide', 'Pred Rest'], yticklabels=['True Slide', 'True Rest'])
        plt.title(f"Detection Stats {year_pair} F{i+1}\n[TP%, FP% / FN%, TN%]")
        plt.savefig(os.path.join(CM_DIR, f"CM_4Dim_{year_pair}_F{i+1}.png")); plt.close()

        # 每折 ROC 和 PR 曲线 ---
        y_true_bin_fold = (y.iloc[te] == 1).astype(int)
        y_prob_fold = model.predict_proba(X.iloc[te])[:, 1]
        
        if len(np.unique(y_true_bin_fold)) > 1:
        # 分折 ROC
            fpr_f, tpr_f, _ = roc_curve(y_true_bin_fold, y_prob_fold)
            plt.figure(figsize=(6, 5)); plt.plot(fpr_f, tpr_f, label=f'AUC={auc(fpr_f, tpr_f):.3f}')
            plt.plot([0, 1], [0, 1], 'k--'); plt.title(f"ROC Fold {i+1} ({year_pair})"); plt.legend()
            plt.savefig(os.path.join(ROC_DIR, f"ROC_{year_pair}_F{i+1}.png")); plt.close()
        # 分折 PR
            prec_f, rec_f, _ = precision_recall_curve(y_true_bin_fold, y_prob_fold)
            plt.figure(figsize=(6, 5)); plt.plot(rec_f, prec_f); plt.title(f"PR Fold {i+1} ({year_pair})")
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.savefig(os.path.join(PR_DIR, f"PR_{year_pair}_F{i+1}.png")); plt.close()
        else:
            log(f"Fold {i+1} ({year_pair}) 跳过 ROC/PR 绘制：该折验证集中无滑坡样本")
    
    # 注释掉 Feature Importance 相关的绘图
    '''
    # 年度特征重要度汇总
    fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': np.mean(fi_list, axis=0)}).sort_values('Importance', ascending=False)
    plt.figure(figsize=(10, 6));
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=fi_df, 
        palette='viridis', 
        hue='Feature', 
        legend=False
    )
    plt.savefig(os.path.join(FI_DIR, f"FI_{year_pair}.png")); plt.close()
    '''

    # 全图预测掩膜
    X_full = df.drop(columns=['label', 'idx', 'block','curvature','aspect'])
    res = np.full(h*w, np.nan); 
    res[df['idx'].values] = np.mean([m.predict_proba(X_full)[:, 1] for m in models], axis=0)
    with rasterio.open(os.path.join(CONFIG["OUTPUT_DIR"], f"Landslide_{year_pair}.tif"), 'w', **profile) as dst:
        dst.write(res.reshape(h, w).astype(np.float32), 1)

    return year_y_true, year_y_prob, year_y_pred, df_train

# =========================================================
# Main 接口
# =========================================================
if __name__ == "__main__":
    g_true, g_prob, g_pred = [], [], []
    all_train_dfs = [] # 用于收集每年的训练集
    
    for yp in ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24","24-25"]:
        log(f"Processing {yp}...")
        y_t, y_p, y_pred, df_t = run_model(yp)
        g_true.extend(y_t); g_prob.extend(y_p); g_pred.extend(y_pred); all_train_dfs.append(df_t)
    
    # 1. 统计全局指标
    calculate_global_metrics(g_true, g_prob, g_pred, all_train_dfs)
    
    # 2. 拼图汇总
    create_summary_grid(CORR_DIR, "Corr_Star_", "Summary_Correlation.png")
    # create_summary_grid(FI_DIR, "FI_", "Summary_FI.png") # 注释掉 Feature Importance 汇总大图的生成
    
    log("All tasks completed.")