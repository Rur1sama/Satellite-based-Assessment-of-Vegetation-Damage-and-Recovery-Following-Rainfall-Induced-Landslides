"""
==============================================================================
Project       : Slide_dect
Module Name   : 加权预测结果.py
File Path     : 代码/机器学习训练/训练主程序/加权预测结果.py
File Created  : 2026-05-27 14:15:21
Description   : 多模型预测结果等权集成计算与制图--结果输出模块
Lines         : 148
Author Note   : 在原“加权预测结果.py”基础上新增：
                1) 四模型输出结果的总混淆矩阵图；
                2) 四模型/集成总 ROC 曲线图；
                3) 四模型/集成总 PR 曲线图；
                4) 总特征相关图；
                5) 总 SHAP 相对贡献与蜂群图源数据。
==============================================================================
"""
from utils.head import *  # 导入基础库、机器学习库、绘图库、SHAP等
from utils.__init__ import *  # 导入 CONFIG、MODEL_FEATURE_DROP_COLUMNS、log等
from utils.base_utils import * # 导入基础处理工具

import re
from catboost import Pool
from sklearn.metrics import average_precision_score

# =========================================================
# 集成特定配置
# =========================================================
ENSEMBLE_OUT_DIR = os.path.join(CONFIG["MODELS_BASE_DIR"], "Ensemble_Final")
os.makedirs(ENSEMBLE_OUT_DIR, exist_ok=True)

# 总评估输出文件夹
ENSEMBLE_EVAL_DIR = os.path.join(ENSEMBLE_OUT_DIR, "Ensemble_Evaluation")
os.makedirs(ENSEMBLE_EVAL_DIR, exist_ok=True)

# SHAP 计算抽样参数：数据量过大时可适当调小
MAX_SHAP_SAMPLE_PER_ALGO_YEAR = 1000
MAX_TOTAL_SHAP_SAMPLES = 8000
MAX_BEESWARM_DISPLAY_SAMPLES = 350
TARGET_CLASS_IDX = 1  # 1 = 滑坡类

# 特征相关图最大抽样量，防止 DataFrame 太大导致绘图慢
MAX_CORR_SAMPLE_SIZE = 50000

# 二分类阈值：用于滑坡/非滑坡混淆矩阵。三分类混淆矩阵仍使用 argmax。
LANDSLIDE_THRESHOLD = 0.8


# =========================================================
# 通用绘图配置
# =========================================================
def set_plot_style():
    """
    统一图件字体和负号显示。
    """
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 一、多模型最终概率栅格等权集成与预览图
# =========================================================
def plot_ensemble_result(ensemble_sum, meta, year_pair, out_path):
    """
    生成并保存带经纬网和比例尺的最终预测概率图。
    """
    set_plot_style() # 设置字体为 SimHei（黑体）并解决负号显示问题
    fig, ax = plt.subplots(figsize=(16, 12))
    # UTM -> 经纬度转换
    transform = meta['transform']
    height, width = ensemble_sum.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    transformer = Transformer.from_crs(meta['crs'], "EPSG:4326", always_xy=True)
    lon_min, lat_max = transformer.transform(left, top)
    lon_max, lat_min = transformer.transform(right, bottom)
    
    # 绘图
    img = ax.imshow(
        ensemble_sum,
        cmap='terrain',
        vmin=0,
        vmax=1,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin='upper'
    )
    # 地理坐标轴样式
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.8, color='gray', alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    # 色标与标题
    cbar = plt.colorbar(img, ax=ax, shrink=0.75)
    cbar.set_label('预测概率', fontsize=12)
    ax.set_title(f"合成后滑坡预测概率图 - {year_pair}", fontsize=18, fontweight='bold')

    # =========================================================
    #  添加 5 km 分段比例尺，每段 1 km
    # =========================================================
    meters_per_degree = 111320 * np.cos(np.deg2rad(lat_min))
    segment_lon = 1000 / meters_per_degree
    bar_lon0 = lon_min + (lon_max - lon_min) * 0.04
    bar_lat0 = lat_min + (lat_max - lat_min) * 0.035
    bar_height = (lat_max - lat_min) * 0.008
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
    for i in range(6):
        x = bar_lon0 + i * segment_lon
        ax.text(x, bar_lat0 - bar_height * 1.8, f"{i}", ha='center', va='top', fontsize=10)
    ax.text(bar_lon0 + 5.3 * segment_lon, bar_lat0 - bar_height * 1.8, "km", fontsize=11)
    
    # 保存结果
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()

def run_ensemble(year_pair):
    '''
    对指定年份对的四张分类器概率图进行等权算术平均。
    仅当四张图均存在且栅格尺寸、空间参考和仿射变换一致时才输出结果。

    :param year_pair: Sentinel-2 年份对
    :return: None
    '''
    log(f"正在处理年份: {year_pair} ...")
    probability_rasters = []
    valid_mask = None
    meta = None

    # 迭代载入各模型结果
    for algo_name in CONFIG["MODELS_INFO"].keys():
        model_path = os.path.join(CONFIG["MODELS_BASE_DIR"], algo_name)
        tif_path = os.path.join(model_path, f"Landslide_{year_pair}.tif")

        if not os.path.exists(tif_path):
            raise FileNotFoundError(
                f"{year_pair} 缺少 {algo_name} 最终概率图: {tif_path}"
            )

        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float32)

            # 记录元数据（以第一个成功读取的模型为准）
            if meta is None:
                meta = src.profile.copy()
                meta.update(dtype=rasterio.float32, nodata=np.nan)
                valid_mask = np.ones_like(data, dtype=bool)
                log(f"  > 统一参考系 CRS: {meta['crs']}")
            else:
                if data.shape != (meta["height"], meta["width"]):
                    raise RuntimeError(
                        f"{algo_name} {year_pair} 概率图尺寸不一致。"
                    )
                if src.crs != meta["crs"]:
                    raise RuntimeError(
                        f"{algo_name} {year_pair} 概率图 CRS 不一致。"
                    )
                if src.transform != meta["transform"]:
                    raise RuntimeError(
                        f"{algo_name} {year_pair} 概率图仿射变换不一致。"
                    )

            invalid = (data < 0) | (np.isnan(data))
            valid_mask &= (~invalid)
            probability_rasters.append(
                np.where(invalid, 0, data)
            )
            log(f"  + 已载入 {algo_name} 最终概率图")

    expected_count = len(CONFIG["MODELS_INFO"])
    if len(probability_rasters) != expected_count:
        raise RuntimeError(
            f"{year_pair} 仅载入 {len(probability_rasters)}/{expected_count} "
            "张分类器概率图。"
        )

    ensemble_mean = np.mean(
        np.stack(probability_rasters, axis=0),
        axis=0
    ).astype(np.float32)
    ensemble_mean[~valid_mask] = np.nan

    out_tif_path = os.path.join(
        ENSEMBLE_OUT_DIR,
        f"Ensemble_Final_{year_pair}.tif"
    )
    with rasterio.open(out_tif_path, 'w', **meta) as dst:
        dst.write(ensemble_mean, 1)

    out_png_path = os.path.join(
        ENSEMBLE_OUT_DIR,
        f"Preview_{year_pair}.png"
    )
    plot_ensemble_result(
        ensemble_mean,
        meta,
        year_pair,
        out_png_path
    )

    log(f"  [Success] {year_pair} 等权集成结果与预览图保存完毕。")
# =========================================================
# 二、复原每个模型的 GroupKFold 折外预测 OOF
# =========================================================
def _get_fold_model_path(model_dir, prefix, year_pair, fold_idx):
    '''
    获取指定年份对和折数对应的模型路径。
    优先匹配精确文件名；若不存在，则搜索同一年份对的折模型并核对折数。

    :param model_dir: 折模型所在目录
    :param prefix: 分类器文件名前缀
    :param year_pair: Sentinel-2 年份对
    :param fold_idx: 交叉验证折序号
    :return: 匹配的模型文件路径；未找到时返回 None
    '''
    direct_path = os.path.join(model_dir, f"{prefix}_{year_pair}_Fold{fold_idx}.pkl")
    if os.path.exists(direct_path):
        return direct_path

    pattern = os.path.join(model_dir, f"{prefix}_{year_pair}_Fold*.pkl")
    candidates = glob.glob(pattern)
    for p in candidates:
        m = re.search(r"Fold(\d+)\.pkl$", os.path.basename(p))
        if m and int(m.group(1)) == fold_idx:
            return p
    return None


def _predict_proba_3class(model, X_part):
    '''
    将分类器输出统一为背景、滑坡和其他扰动三列概率。
    根据模型的 classes_ 属性恢复类别顺序，避免概率列错配。

    :param model: 已训练的三分类模型
    :param X_part: 待预测的特征表
    :return: 按类别 0、1、2 排列的概率矩阵
    '''
    raw_proba = model.predict_proba(X_part)
    proba_3 = np.zeros((len(X_part), 3), dtype=np.float32)

    classes = getattr(model, "classes_", np.array([0, 1, 2]))
    for j, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int <= 2:
            proba_3[:, cls_int] = raw_proba[:, j]

    return proba_3


def compute_oof_predictions_for_algo_year(algo_name, prefix, year_pair):
    '''
    根据共享训练样本中的固定折编号复原一个分类器的年度 OOF 预测。
    每个折模型只预测该折对应的验证样本，并将结果写回原样本位置。

    :param algo_name: 分类器名称
    :param prefix: 分类器文件名前缀
    :param year_pair: Sentinel-2 年份对
    :return: 包含样本标识、真实标签和三分类 OOF 概率的数据表
    '''
    base_dir = os.path.join(CONFIG["MODELS_BASE_DIR"], algo_name)
    model_dir = os.path.join(base_dir, "Saved_Models")
    data_path = os.path.join(CONFIG["SHARED_TRAIN_DIR"], f"train_{year_pair}.csv")

    if not os.path.exists(data_path):
        log(f"  [跳过] {algo_name} {year_pair}: 找不到训练CSV {data_path}")
        return None

    df = pd.read_csv(data_path)
    if df.empty:
        log(f"  [跳过] {algo_name} {year_pair}: 训练CSV为空")
        return None

    required_cols = {"idx", "label", "block", "fold_id"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"{data_path} 缺少必要字段: {sorted(missing_cols)}"
        )

    X = df.drop(columns=MODEL_FEATURE_DROP_COLUMNS, errors='ignore')
    y = df["label"].astype(int).reset_index(drop=True)
    # 直接读取共享训练CSV中已经固定的fold编号
    fold_ids = df["fold_id"].astype(int).to_numpy()

    n = len(df)
    proba_oof = np.full((n, 3), np.nan, dtype=np.float32)
    pred_oof = np.full(n, -1, dtype=np.int16)

    # 按训练时保存的fold编号复原每一折验证集
    for fold_idx in range(1, CONFIG["N_SPLITS"] + 1):
        te = np.flatnonzero(fold_ids == fold_idx)
        if len(te) == 0:
            raise RuntimeError(
                f"{algo_name} {year_pair} Fold{fold_idx} 没有验证样本。"
            )

        model_path = _get_fold_model_path(model_dir, prefix, year_pair, fold_idx)
        if model_path is None:
            raise FileNotFoundError(
                f"找不到模型: {prefix}_{year_pair}_Fold{fold_idx}.pkl"
            )

        model = joblib.load(model_path)
        X_te = X.iloc[te]
        proba_3 = _predict_proba_3class(model, X_te)
        pred = np.argmax(proba_3, axis=1).astype(np.int16)

        proba_oof[te, :] = proba_3
        pred_oof[te] = pred

        del model
        gc.collect()

    valid = np.isfinite(proba_oof).all(axis=1) & (pred_oof >= 0)
    if not np.all(valid):
        raise RuntimeError(
            f"{algo_name} {year_pair} 仅复原了 {valid.sum()}/{n} 个 OOF 预测。"
        )

    out = pd.DataFrame({
        "algo_name": algo_name,
        "year_pair": year_pair,
        "idx": df["idx"].values,
        "label": y.values,
        "block": df["block"].values,
        "fold_id": fold_ids,
        "pred": pred_oof,
        "prob_0": proba_oof[:, 0],
        "prob_1": proba_oof[:, 1],
        "prob_2": proba_oof[:, 2],
    })
    return out

def collect_all_oof_predictions():
    '''
    收集四个分类器在全部年份对上的 OOF 预测。
    每个分类器的年度结果按行合并，并保存可复核的中间数据表。

    :return: 以分类器名称为键、OOF 预测表为值的字典
    '''
    all_oof = {}

    for algo_name, prefix in CONFIG["MODELS_INFO"].items():
        algo_tables = []
        log(f"开始复原 {algo_name} 的折外预测...")

        for yp in CONFIG["YEAR_PAIRS"]:
            table = compute_oof_predictions_for_algo_year(algo_name, prefix, yp)
            if table is not None and not table.empty:
                algo_tables.append(table)

        if algo_tables:
            all_oof[algo_name] = pd.concat(algo_tables, ignore_index=True)
            out_csv = os.path.join(ENSEMBLE_EVAL_DIR, f"OOF_{algo_name.replace(' ', '_')}.csv")
            all_oof[algo_name].to_csv(out_csv, index=False, encoding="utf-8-sig")
            log(f"  [完成] {algo_name}: 折外预测样本数 = {len(all_oof[algo_name])}")
        else:
            log(f"  [跳过] {algo_name}: 没有可用折外预测")

    return all_oof

def build_equal_weight_ensemble_oof(all_oof):
    '''
    对完全对齐的四分类器 OOF 概率进行等权软投票。
    四个分类器必须包含相同的样本、标签、空间块和固定折编号；不允许静默取交集。

    :param all_oof: 四个分类器在全部年份对上的 OOF 预测字典
    :return: 包含等权集成概率和预测类别的 OOF 数据表
    '''
    if not all_oof:
        return None

    expected_algos = list(CONFIG["MODELS_INFO"].keys())
    missing_algos = [name for name in expected_algos if name not in all_oof]
    if missing_algos:
        raise RuntimeError(
            f"缺少分类器 OOF 结果: {missing_algos}"
        )

    merged = None
    key_cols = ["year_pair", "idx", "label", "block", "fold_id"]
    expected_n = None

    for algo_name in expected_algos:
        df = all_oof[algo_name]
        if df.duplicated(key_cols).any():
            raise RuntimeError(
                f"{algo_name} OOF 结果存在重复样本键。"
            )

        if expected_n is None:
            expected_n = len(df)
        elif len(df) != expected_n:
            raise RuntimeError(
                f"四分类器 OOF 样本量不一致: expected={expected_n}, "
                f"{algo_name}={len(df)}"
            )

        tmp = df[key_cols + ["prob_0", "prob_1", "prob_2"]].copy()
        safe_name = algo_name.replace(" ", "_")
        tmp = tmp.rename(columns={
            "prob_0": f"prob_0__{safe_name}",
            "prob_1": f"prob_1__{safe_name}",
            "prob_2": f"prob_2__{safe_name}",
        })

        if merged is None:
            merged = tmp
        else:
            merged = pd.merge(
                merged,
                tmp,
                on=key_cols,
                how="inner",
                validate="one_to_one"
            )

    if merged is None or merged.empty:
        raise RuntimeError("四分类器 OOF 结果无法完成逐样本对齐。")

    if len(merged) != expected_n:
        raise RuntimeError(
            "四分类器 OOF 样本键不完全一致: "
            f"expected={expected_n}, aligned={len(merged)}"
        )

    probability_tables = []
    for algo_name in expected_algos:
        safe_name = algo_name.replace(" ", "_")
        cols = [f"prob_0__{safe_name}", f"prob_1__{safe_name}", f"prob_2__{safe_name}"]
        probability_tables.append(
            merged[cols].to_numpy(dtype=np.float32)
        )

    ens_prob = np.mean(
        np.stack(probability_tables, axis=0),
        axis=0
    )
    merged["prob_0"] = ens_prob[:, 0]
    merged["prob_1"] = ens_prob[:, 1]
    merged["prob_2"] = ens_prob[:, 2]
    merged["pred"] = np.argmax(ens_prob, axis=1).astype(np.int16)
    merged["algo_name"] = "Equal-weight Ensemble"

    out_csv = os.path.join(ENSEMBLE_EVAL_DIR, "OOF_EqualWeight_Ensemble.csv")
    merged[["algo_name", *key_cols, "pred", "prob_0", "prob_1", "prob_2"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    log(f"[完成] 四分类器等权集成 OOF 样本数 = {len(merged)}")
    return merged

# =========================================================
# 三、总混淆矩阵、ROC、PR绘图
# =========================================================
def _binary_detection_arrays(df):
    '''
    将三分类标签和概率转换为滑坡与非滑坡二分类评价数组。

    :param df: 包含 label 和 prob_1 字段的 OOF 预测表
    :return: 二分类真实标签、阈值预测标签和滑坡概率
    '''
    y_true_bin = (df["label"].values.astype(int) == 1).astype(int)
    y_prob = df["prob_1"].values.astype(float)
    y_pred_bin = (y_prob >= LANDSLIDE_THRESHOLD).astype(int)
    return y_true_bin, y_pred_bin, y_prob


def _row_normalize_confusion_matrix(cm):
    '''
    按真实类别所在行对混淆矩阵进行归一化。

    :param cm: 未归一化的混淆矩阵
    :return: 每行之和为 1 的浮点混淆矩阵
    '''
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm),
        where=row_sums != 0
    )


def _safe_metric_ratio(numerator, denominator):
    '''
    安全计算评价指标中的比值，避免分母为零时产生无穷值。

    :param numerator: 指标分子
    :param denominator: 指标分母
    :return: 有效比值；分母为零时返回 NaN
    '''
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _interpolate_pr_for_display(recall_values, precision_values, recall_grid):
    '''
    将原始 PR 曲线插值到统一 Recall 网格，仅用于 Origin 绘图源数据。
    对相同 Recall 的竖直曲线段保留最高 Precision；AP 仍由原始 OOF
    预测直接计算，不使用插值后的曲线。

    :param recall_values: 原始 PR 曲线的 Recall 坐标
    :param precision_values: 原始 PR 曲线的 Precision 坐标
    :param recall_grid: 0–1 的统一 Recall 绘图网格
    :return: 与统一 Recall 网格对应的 Precision 数组
    '''
    recall_values = np.asarray(recall_values, dtype=float)
    precision_values = np.asarray(precision_values, dtype=float)
    order = np.argsort(recall_values, kind="mergesort")
    recall_sorted = recall_values[order]
    precision_sorted = precision_values[order]

    unique_recall, start_indices = np.unique(
        recall_sorted,
        return_index=True
    )
    precision_at_unique_recall = np.maximum.reduceat(
        precision_sorted,
        start_indices
    )
    return np.interp(
        recall_grid,
        unique_recall,
        precision_at_unique_recall
    )


def _write_evaluation_source_workbook(sheets, output_path):
    '''
    将全局 OOF 评价曲线、指标和混淆矩阵写入同一个格式化 Excel。

    :param sheets: 以工作表名称为键、DataFrame 为值的数据字典
    :param output_path: Excel 输出路径
    :return: None
    '''
    from openpyxl.styles import Alignment, Font, PatternFill

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False

            for cell in worksheet[1]:
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.font = Font(color="FFFFFF", bold=True, name="Arial")
                cell.alignment = Alignment(horizontal="center")

            for column_cells in worksheet.columns:
                header = str(column_cells[0].value or "")
                observed_width = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells[:200]
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    max(len(header) + 2, observed_width + 2, 12),
                    34
                )


def export_global_evaluation_source_data(all_oof, ensemble_df):
    '''
    导出七个年份对合并后的全局 OOF 评价源数据。
    ROC 和 PR 曲线使用 0–1、步长 0.001 的公共坐标网格，便于在 Origin
    中直接比较四个分类器和等权集成；AUC、AP 与阈值指标均从完整 OOF
    预测计算。工作簿同时提供三分类和二分类混淆矩阵长表。

    :param all_oof: 以分类器名称为键、OOF 预测表为值的字典
    :param ensemble_df: 完全对齐并完成等权概率集成的 OOF 预测表
    :return: 已输出的 Excel 文件路径
    '''
    if not all_oof:
        raise ValueError("没有可用于导出的单模型 OOF 预测。")

    model_tables = list(all_oof.items())
    if ensemble_df is not None and not ensemble_df.empty:
        model_tables.append(("Equal-weight Ensemble", ensemble_df))

    curve_grid = np.linspace(0.0, 1.0, 1001)
    roc_columns = {}
    pr_columns = {}
    metric_rows = []
    cm_three_class_rows = []
    cm_binary_rows = []
    three_class_labels = [0, 1, 2]
    three_class_names = ["Background", "Landslide", "Disturbance"]

    for model_name, model_df in model_tables:
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
        y_true_bin, y_pred_bin, y_prob = _binary_detection_arrays(model_df)
        if len(np.unique(y_true_bin)) < 2:
            log(f"  [跳过] {model_name}: 二分类真实标签不足两个类别。")
            continue

        fpr_values, tpr_values, _ = roc_curve(y_true_bin, y_prob)
        precision_values, recall_values, _ = precision_recall_curve(
            y_true_bin,
            y_prob
        )
        tpr_grid = np.interp(curve_grid, fpr_values, tpr_values)
        precision_grid = _interpolate_pr_for_display(
            recall_values,
            precision_values,
            curve_grid
        )

        roc_columns[f"FPR_{safe_name}"] = curve_grid
        roc_columns[f"TPR_{safe_name}"] = tpr_grid
        roc_columns[f"FPR_{safe_name}_percent"] = curve_grid * 100.0
        roc_columns[f"TPR_{safe_name}_percent"] = tpr_grid * 100.0
        pr_columns[f"Recall_{safe_name}"] = curve_grid
        pr_columns[f"Precision_{safe_name}"] = precision_grid
        pr_columns[f"Recall_{safe_name}_percent"] = curve_grid * 100.0
        pr_columns[f"Precision_{safe_name}_percent"] = precision_grid * 100.0

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin,
            y_pred_bin,
            labels=[0, 1]
        ).ravel()
        binary_precision = _safe_metric_ratio(tp, tp + fp)
        binary_recall = _safe_metric_ratio(tp, tp + fn)
        binary_specificity = _safe_metric_ratio(tn, tn + fp)
        metric_rows.append({
            "model": model_name,
            "n_samples": int(len(y_true_bin)),
            "n_landslide": int(y_true_bin.sum()),
            "n_non_landslide": int((y_true_bin == 0).sum()),
            "prevalence_percent": float(y_true_bin.mean() * 100.0),
            "roc_auc": float(auc(fpr_values, tpr_values)),
            "average_precision": float(
                average_precision_score(y_true_bin, y_prob)
            ),
            "classification_threshold": float(LANDSLIDE_THRESHOLD),
            "accuracy_at_threshold": _safe_metric_ratio(
                tp + tn,
                len(y_true_bin)
            ),
            "precision_at_threshold": binary_precision,
            "recall_at_threshold": binary_recall,
            "specificity_at_threshold": binary_specificity,
            "f1_at_threshold": _safe_metric_ratio(
                2.0 * binary_precision * binary_recall,
                binary_precision + binary_recall
            ) if np.isfinite(binary_precision) and np.isfinite(binary_recall) else np.nan
        })

        cm_three = confusion_matrix(
            model_df["label"].values.astype(int),
            model_df["pred"].values.astype(int),
            labels=three_class_labels
        ).astype(int)
        cm_three_fraction = _row_normalize_confusion_matrix(cm_three)
        for actual_idx, actual_name in enumerate(three_class_names):
            for predicted_idx, predicted_name in enumerate(three_class_names):
                cm_three_class_rows.append({
                    "model": model_name,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "count": int(cm_three[actual_idx, predicted_idx]),
                    "row_fraction": float(
                        cm_three_fraction[actual_idx, predicted_idx]
                    ),
                    "row_percent": float(
                        cm_three_fraction[actual_idx, predicted_idx] * 100.0
                    )
                })

        cm_binary = np.array([[tn, fp], [fn, tp]], dtype=int)
        cm_binary_fraction = _row_normalize_confusion_matrix(cm_binary)
        binary_names = ["Non-landslide", "Landslide"]
        for actual_idx, actual_name in enumerate(binary_names):
            for predicted_idx, predicted_name in enumerate(binary_names):
                cm_binary_rows.append({
                    "model": model_name,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "count": int(cm_binary[actual_idx, predicted_idx]),
                    "row_fraction": float(
                        cm_binary_fraction[actual_idx, predicted_idx]
                    ),
                    "row_percent": float(
                        cm_binary_fraction[actual_idx, predicted_idx] * 100.0
                    ),
                    "classification_threshold": float(LANDSLIDE_THRESHOLD)
                })

    settings = pd.DataFrame({
        "item": [
            "evaluation_scope",
            "positive_class",
            "curve_grid_min",
            "curve_grid_max",
            "curve_grid_step",
            "classification_threshold",
            "roc_metric_source",
            "pr_metric_source",
            "curve_interpolation_note"
        ],
        "value": [
            "Pooled out-of-fold predictions across seven year pairs",
            "Landslide (label = 1)",
            0.0,
            1.0,
            0.001,
            LANDSLIDE_THRESHOLD,
            "Exact ROC-AUC from all pooled OOF predictions",
            "Exact average precision from all pooled OOF predictions",
            "The 1001-point curves are display coordinates only; metrics are not recalculated from interpolation."
        ]
    })

    output_path = os.path.join(
        ENSEMBLE_EVAL_DIR,
        "Global_OOF_Evaluation_Source_Data.xlsx"
    )
    _write_evaluation_source_workbook({
        "PR_Curves": pd.DataFrame(pr_columns),
        "ROC_Curves": pd.DataFrame(roc_columns),
        "Metrics": pd.DataFrame(metric_rows),
        "CM_3Class": pd.DataFrame(cm_three_class_rows),
        "CM_Binary": pd.DataFrame(cm_binary_rows),
        "Settings": settings
    }, output_path)
    log(f"[完成] 全局 OOF 评价 Excel 源数据已输出: {output_path}")
    return output_path


def plot_four_model_cm_summary(all_oof):
    '''
    输出四个单模型的总混淆矩阵图：
    1) 三分类 9格 CM，一张2x2总图；
    2) 滑坡/非滑坡 4格 CM，一张2x2总图。

    :param all_oof: 以分类器名称为键、OOF 预测表为值的字典
    :return: None
    '''
    set_plot_style()
    if not all_oof:
        return

    # ---------- 三分类混淆矩阵 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    labels = [0, 1, 2]
    class_names = ["Background", "Slide", "Disturb"]

    for ax, (algo_name, df) in zip(axes, all_oof.items()):
        cm = confusion_matrix(df["label"], df["pred"], labels=labels).astype(float)
        cm_perc = _row_normalize_confusion_matrix(cm)
        sns.heatmap(
            cm_perc,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            cbar=False,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names
        )
        ax.set_title(f"{algo_name} - 3-Class CM")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(all_oof):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(ENSEMBLE_EVAL_DIR, "Summary_Four_Model_CM_9Classes.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # ---------- 二分类检测混淆矩阵 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (algo_name, df) in zip(axes, all_oof.items()):
        y_true_bin, y_pred_bin, _ = _binary_detection_arrays(df)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        cm_4dim = _row_normalize_confusion_matrix(
            np.array([
                [tp, fn],
                [fp, tn]
            ])
        )
        sns.heatmap(
            cm_4dim,
            annot=True,
            fmt='.2%',
            cmap='Greens',
            cbar=False,
            ax=ax,
            xticklabels=['Pred Slide', 'Pred Rest'],
            yticklabels=['True Slide', 'True Rest']
        )
        acc = (tp + tn) / max(len(y_true_bin), 1)
        ax.set_title(f"{algo_name} - Detection CM\nAcc={acc:.2%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for ax in axes[len(all_oof):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(ENSEMBLE_EVAL_DIR, "Summary_Four_Model_CM_4Dimensions.png"), dpi=600, bbox_inches='tight')
    plt.close()


def plot_ensemble_cm(ensemble_df):
    '''
    输出七个年份对合并后的等权集成三分类混淆矩阵。
    先根据四分类器平均概率确定预测类别，再按真实类别逐行归一化。

    :param ensemble_df: 完全对齐并完成等权概率集成的 OOF 预测表
    :return: None
    '''
    set_plot_style()
    if ensemble_df is None or ensemble_df.empty:
        return

    labels = [0, 1, 2]
    class_names = ["Background", "Landslide", "Disturbance"]
    cm = confusion_matrix(
        ensemble_df["label"],
        ensemble_df["pred"],
        labels=labels
    ).astype(float)
    cm_perc = _row_normalize_confusion_matrix(cm)

    pd.DataFrame(
        cm.astype(int),
        index=class_names,
        columns=class_names
    ).to_csv(
        os.path.join(ENSEMBLE_EVAL_DIR, "Ensemble_Global_CM_Counts.csv"),
        encoding="utf-8-sig"
    )
    pd.DataFrame(
        cm_perc,
        index=class_names,
        columns=class_names
    ).to_csv(
        os.path.join(ENSEMBLE_EVAL_DIR, "Ensemble_Global_CM_RowNormalized.csv"),
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm_perc,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        vmin=0,
        vmax=1,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Equal-weight ensemble: pooled OOF predictions")
    plt.xlabel("Predicted class")
    plt.ylabel("Observed class")
    plt.tight_layout()
    plt.savefig(
        os.path.join(ENSEMBLE_EVAL_DIR, "Ensemble_Global_CM.png"),
        dpi=600,
        bbox_inches='tight'
    )
    plt.close()


def plot_global_roc_pr_curves(all_oof, ensemble_df):
    '''
    输出总 ROC 曲线图和总 PR 曲线图。
    每张图同时展示四个单模型曲线和等权集成曲线，所有结果均来自合并年份后的 OOF 预测。

    :param all_oof: 以分类器名称为键、OOF 预测表为值的字典
    :param ensemble_df: 完全对齐并完成等权概率集成的 OOF 预测表
    :return: None
    '''
    set_plot_style()

    # ---------- ROC ----------
    plt.figure(figsize=(9, 7))
    for algo_name, df in all_oof.items():
        y_true_bin, _, y_prob = _binary_detection_arrays(df)
        if len(np.unique(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{algo_name} AUC={roc_auc:.3f}")

    if ensemble_df is not None and not ensemble_df.empty:
        y_true_bin, _, y_prob = _binary_detection_arrays(ensemble_df)
        if len(np.unique(y_true_bin)) >= 2:
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=3, linestyle='--', label=f"Equal-weight ensemble AUC={roc_auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle=':', lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pooled OOF ROC curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(ENSEMBLE_EVAL_DIR, "Global_ROC_FourModels_Ensemble.png"), dpi=600, bbox_inches='tight')
    plt.close()

    # ---------- PR ----------
    plt.figure(figsize=(9, 7))
    for algo_name, df in all_oof.items():
        y_true_bin, _, y_prob = _binary_detection_arrays(df)
        if len(np.unique(y_true_bin)) < 2:
            continue
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_prob)
        ap = average_precision_score(y_true_bin, y_prob)
        plt.plot(recall_vals, precision_vals, lw=2, label=f"{algo_name} AP={ap:.3f}")

    if ensemble_df is not None and not ensemble_df.empty:
        y_true_bin, _, y_prob = _binary_detection_arrays(ensemble_df)
        if len(np.unique(y_true_bin)) >= 2:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true_bin, y_prob)
            ap = average_precision_score(y_true_bin, y_prob)
            plt.plot(recall_vals, precision_vals, lw=3, linestyle='--', label=f"Equal-weight ensemble AP={ap:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pooled OOF precision-recall curves")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(ENSEMBLE_EVAL_DIR, "Global_PR_FourModels_Ensemble.png"), dpi=600, bbox_inches='tight')
    plt.close()


# =========================================================
# 四、总特征相关图
# =========================================================
def _safe_corr_annotation(df_features, corr):
    """
    为相关性热力图生成标注矩阵：
    下三角为 Pearson r，上三角为显著性星号，对角线为 1.00。
    """
    annot_matrix = corr.copy().astype(str)
    cols = list(df_features.columns)

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            if i == j:
                annot_matrix.iloc[i, j] = "1.00"
                continue

            x = df_features[col_i]
            y = df_features[col_j]
            valid = np.isfinite(x) & np.isfinite(y)

            if valid.sum() < 3 or x[valid].nunique() <= 1 or y[valid].nunique() <= 1:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(x[valid], y[valid])

            if i < j:
                if not np.isfinite(p):
                    star = ""
                elif p < 0.001:
                    star = "***"
                elif p < 0.01:
                    star = "**"
                elif p < 0.05:
                    star = "*"
                else:
                    star = ""
                annot_matrix.iloc[i, j] = star
            else:
                annot_matrix.iloc[i, j] = "" if not np.isfinite(r) else f"{r:.2f}"

    return annot_matrix


def collect_training_feature_tables():
    '''
    汇总七个年份对的共享训练样本，用于绘制总体特征相关图。
    每个年度样本只读取一次，避免因四分类器复用而重复计数。

    :return: 合并年份标识后的共享训练特征表；无数据时返回 None
    '''
    tables = []

    for year_pair in CONFIG["YEAR_PAIRS"]:
        data_path = os.path.join(
            CONFIG["SHARED_TRAIN_DIR"],
            f"train_{year_pair}.csv"
        )
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            if not df.empty:
                df["source_year_pair"] = year_pair
                tables.append(df)

    if not tables:
        return None

    full_df = pd.concat(tables, ignore_index=True)
    return full_df


def plot_total_feature_correlation():
    '''
    输出所有模型、所有年份对训练样本合并后的总特征相关图。

    :return: None
    '''
    set_plot_style()
    df = collect_training_feature_tables()
    if df is None or df.empty:
        log("[跳过] 没有找到训练CSV，无法绘制总特征相关图。")
        return

    features_to_plot = df.drop(
        columns=[
            'label', 'idx', 'block', 'fold_id', 'curvature',
            'aspect', 'source_year_pair'
        ],
        errors='ignore'
    )
    features_to_plot = features_to_plot.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()

    if features_to_plot.empty:
        log("[跳过] 总特征表为空，无法绘制总特征相关图。")
        return

    if len(features_to_plot) > MAX_CORR_SAMPLE_SIZE:
        features_to_plot = features_to_plot.sample(MAX_CORR_SAMPLE_SIZE, random_state=42)

    corr = features_to_plot.corr()
    annot_matrix = _safe_corr_annotation(features_to_plot, corr)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=annot_matrix, fmt="", cmap='RdBu', center=0, square=True)
    plt.title("Global Feature Correlation - Shared Samples / All Years")
    plt.savefig(os.path.join(ENSEMBLE_EVAL_DIR, "Global_Feature_Correlation_AllModels.png"), dpi=600, bbox_inches='tight')
    plt.close()
    log("[完成] 总特征相关图已输出。")


# =========================================================
# 五、总 SHAP 解释源数据
# =========================================================
def _extract_class_shap_values(values, class_idx=TARGET_CLASS_IDX):
    '''
    将不同模型、不同 SHAP 版本返回的 values 统一转成二维数组：[n_samples, n_features]。

    :param values: SHAP 解释器返回的原始贡献数组
    :param class_idx: 需要提取的类别索引
    :return: 指定类别的二维 SHAP 贡献矩阵
    '''
    if isinstance(values, list):
        # 老版本 shap 对多分类可能返回 list[class]，每个元素为 [n_samples, n_features]
        return np.asarray(values[class_idx], dtype=np.float32)

    arr = np.asarray(values)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3:
        # 常见格式1：[n_samples, n_features, n_classes]
        if arr.shape[-1] >= class_idx + 1:
            return arr[:, :, class_idx].astype(np.float32)
        # 常见格式2：[n_samples, n_classes, n_features]
        if arr.shape[1] >= class_idx + 1:
            return arr[:, class_idx, :].astype(np.float32)

    raise ValueError(f"无法识别的 SHAP values 维度: {arr.shape}")


def _compute_catboost_shap_values(model_path, X_df):
    '''
    CatBoost 原生 SHAP 计算，并提取滑坡类 SHAP 值。

    :param model_path: CatBoost 折模型文件路径
    :param X_df: 该折对应的验证特征表
    :return: 滑坡类别的二维 SHAP 贡献矩阵
    '''
    model = joblib.load(model_path)
    pool = Pool(X_df)
    shap_raw = np.asarray(model.get_feature_importance(data=pool, type="ShapValues", thread_count=-1, verbose=False))

    if shap_raw.ndim == 3:
        # 兼容两种多分类输出格式
        if shap_raw.shape[1] == X_df.shape[1] + 1:
            values = shap_raw[:, :-1, TARGET_CLASS_IDX]
        else:
            values = shap_raw[:, TARGET_CLASS_IDX, :-1]
    elif shap_raw.ndim == 2:
        values = shap_raw[:, :-1]
    else:
        raise ValueError(f"无法识别的 CatBoost SHAP 维度: {shap_raw.shape}")

    del model, pool, shap_raw
    gc.collect()
    return values.astype(np.float32)


def _compute_tree_shap_values(model_path, X_df):
    '''
    LightGBM、XGBoost、RandomForest 的 TreeExplainer SHAP 计算。

    :param model_path: 树模型文件路径
    :param X_df: 该折对应的验证特征表
    :return: 滑坡类别的二维 SHAP 贡献矩阵
    '''
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    exp = explainer(X_df, check_additivity=False)
    values = _extract_class_shap_values(exp.values, TARGET_CLASS_IDX)

    del explainer, model, exp
    gc.collect()
    return values.astype(np.float32)

def compute_shap_for_algo_year(algo_name, prefix, year_pair):
    '''
    计算一个分类器和年份对的 OOF SHAP 结果。
    每个折模型仅解释该折验证样本；十折结果拼接后计算归一化相对贡献。

    :param algo_name: 分类器名称
    :param prefix: 分类器文件名前缀
    :param year_pair: Sentinel-2 年份对
    :return: 归一化 SHAP 值、对应验证特征表和变量相对贡献
    '''
    base_dir = os.path.join(CONFIG["MODELS_BASE_DIR"], algo_name)
    model_dir = os.path.join(base_dir, "Saved_Models")
    data_path = os.path.join(
        CONFIG["SHARED_TRAIN_DIR"],
        f"train_{year_pair}.csv"
    )

    if not os.path.exists(data_path):
        log(f"  [SHAP跳过] 找不到共享训练样本: {data_path}")
        return None, None, None

    df = pd.read_csv(data_path)
    if df.empty:
        log(f"  [SHAP跳过] 共享训练样本为空: {data_path}")
        return None, None, None

    required_cols = {"fold_id", "label", "block", "idx"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise RuntimeError(
            f"{data_path} 缺少 SHAP 所需字段: {sorted(missing_cols)}"
        )

    X = df.drop(columns=MODEL_FEATURE_DROP_COLUMNS, errors='ignore')
    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    fold_ids = df["fold_id"].astype(int).to_numpy()

    n_per_fold = max(
        1,
        int(np.ceil(
            MAX_SHAP_SAMPLE_PER_ALGO_YEAR / CONFIG["N_SPLITS"]
        ))
    )

    shap_values_list = []
    feature_tables = []

    for fold_idx in tqdm(
        range(1, CONFIG["N_SPLITS"] + 1),
        desc=f"OOF SHAP {algo_name} {year_pair}",
        leave=False
    ):
        model_path = _get_fold_model_path(
            model_dir,
            prefix,
            year_pair,
            fold_idx
        )
        if model_path is None:
            raise FileNotFoundError(
                f"找不到 SHAP 折模型: {prefix}_{year_pair}_Fold{fold_idx}.pkl"
            )

        te = np.flatnonzero(fold_ids == fold_idx)
        if len(te) == 0:
            raise RuntimeError(
                f"{algo_name} {year_pair} Fold{fold_idx} 没有 SHAP 验证样本。"
            )

        X_fold = X.iloc[te]
        if len(X_fold) > n_per_fold:
            X_fold = X_fold.sample(
                n=n_per_fold,
                random_state=CONFIG["RANDOM_STATE"] + fold_idx
            )

        try:
            if algo_name == "CatBoost":
                vals = _compute_catboost_shap_values(model_path, X_fold)
            else:
                vals = _compute_tree_shap_values(model_path, X_fold)

            if vals.shape != X_fold.shape:
                raise RuntimeError(
                    f"SHAP 维度 {vals.shape} 与特征维度 {X_fold.shape} 不一致。"
                )

            shap_values_list.append(vals)
            feature_tables.append(X_fold.reset_index(drop=True))
        except Exception as e:
            raise RuntimeError(
                f"{algo_name} {year_pair} Fold{fold_idx} SHAP 计算失败: {e}"
            ) from e

    if not shap_values_list:
        return None, None, None

    shap_values = np.vstack(shap_values_list).astype(np.float32)
    X_oof = pd.concat(feature_tables, ignore_index=True)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    total_importance = float(mean_abs.sum())
    if not np.isfinite(total_importance) or total_importance <= 0:
        raise RuntimeError(
            f"{algo_name} {year_pair} 的 SHAP 总贡献无效。"
        )

    relative_importance = (
        mean_abs / total_importance
    ).astype(np.float32)
    normalized_values = (
        shap_values / total_importance
    ).astype(np.float32)

    return normalized_values, X_oof, relative_importance


def _sample_shap_display_rows(
    shap_values,
    feature_table,
    sample_meta,
    max_samples=MAX_BEESWARM_DISPLAY_SAMPLES
):
    '''
    从各算法年份组合中近等量抽取固定数量的SHAP展示样本。
    抽样仅影响蜂群图和逐点展示表，不参与总体相对贡献计算。

    :param shap_values: 汇总后的归一化SHAP值矩阵
    :param feature_table: 与SHAP矩阵逐行对应的特征值表
    :param sample_meta: 每个样本对应的算法和年份对信息
    :param max_samples: 蜂群图最多展示的样本数
    :return: 分层抽样后的SHAP矩阵、特征表和样本元数据
    '''
    shap_values = np.asarray(shap_values, dtype=np.float32)
    n_samples = len(feature_table)
    if shap_values.shape[0] != n_samples or len(sample_meta) != n_samples:
        raise RuntimeError("SHAP展示抽样的数据长度不一致。")
    if max_samples <= 0:
        raise ValueError("蜂群图展示样本数必须大于0。")
    if n_samples <= max_samples:
        return (
            shap_values.copy(),
            feature_table.reset_index(drop=True).copy(),
            sample_meta.reset_index(drop=True).copy()
        )

    group_columns = ["algorithm", "year_pair"]
    missing_columns = set(group_columns).difference(sample_meta.columns)
    if missing_columns:
        raise RuntimeError(
            f"SHAP样本元数据缺少分层字段: {sorted(missing_columns)}"
        )

    rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
    group_indices = []
    grouped = sample_meta.groupby(group_columns, sort=True, observed=True)
    for _, group in grouped:
        indices = group.index.to_numpy(dtype=int).copy()
        rng.shuffle(indices)
        group_indices.append(indices)

    selected = []
    cursor = np.zeros(len(group_indices), dtype=int)
    while len(selected) < max_samples:
        added = False
        for group_idx, indices in enumerate(group_indices):
            if cursor[group_idx] >= len(indices):
                continue
            selected.append(indices[cursor[group_idx]])
            cursor[group_idx] += 1
            added = True
            if len(selected) == max_samples:
                break
        if not added:
            break

    selected = np.asarray(selected, dtype=int)
    return (
        shap_values[selected],
        feature_table.iloc[selected].reset_index(drop=True),
        sample_meta.iloc[selected].reset_index(drop=True)
    )


def _build_shap_beeswarm_source(
    shap_values,
    feature_table,
    sample_meta,
    global_relative
):
    '''
    将SHAP蜂群图展示样本转换为可供Excel或Origin重绘的逐点长表。
    `beeswarm_y`复现SHAP 0.49.1原生dot蜂群的100分箱对称堆叠；14个变量
    的中心按旧图规则编号为1–14，蜂群点在各整数中心附近展开。颜色值按每个
    变量的第5和第95百分位截断并缩放至0–1。

    :param shap_values: 抽样后的归一化SHAP值矩阵
    :param feature_table: 与SHAP矩阵逐行对应的特征值表
    :param sample_meta: 每个样本对应的算法和年份对信息
    :param global_relative: 各变量的总体相对贡献
    :return: SHAP蜂群图逐点长表
    '''
    shap_values = np.asarray(shap_values, dtype=np.float32)
    global_relative = np.asarray(global_relative, dtype=np.float32)
    n_samples, n_features = shap_values.shape

    if feature_table.shape != (n_samples, n_features):
        raise RuntimeError("SHAP矩阵与特征表维度不一致。")
    if len(sample_meta) != n_samples:
        raise RuntimeError("SHAP样本与算法年份元数据长度不一致。")
    if len(global_relative) != n_features:
        raise RuntimeError("SHAP总体贡献与特征数量不一致。")

    descending_order = np.argsort(-global_relative)
    feature_rank = np.empty(n_features, dtype=np.int16)
    feature_rank[descending_order] = np.arange(1, n_features + 1)
    sample_ids = np.arange(1, n_samples + 1, dtype=np.int32)
    feature_order = np.argsort(global_relative)
    rng = np.random.default_rng(CONFIG["RANDOM_STATE"])
    n_bins = 100
    row_height = 0.4
    frames = []

    for feature_y, feature_idx in enumerate(feature_order):
        feature = feature_table.columns[feature_idx]
        feature_center = feature_y + 1.0
        plot_order = np.arange(n_samples)
        rng.shuffle(plot_order)

        shap_x = shap_values[plot_order, feature_idx].astype(float)
        feature_value = pd.to_numeric(
            feature_table.iloc[:, feature_idx],
            errors="coerce"
        ).to_numpy(dtype=float)[plot_order]

        finite_feature = np.isfinite(feature_value)
        scaled_feature = np.full(n_samples, np.nan, dtype=float)
        if finite_feature.any():
            feature_min = float(np.nanpercentile(feature_value, 5))
            feature_max = float(np.nanpercentile(feature_value, 95))
            if feature_min == feature_max:
                feature_min = float(np.nanpercentile(feature_value, 1))
                feature_max = float(np.nanpercentile(feature_value, 99))
            if feature_min == feature_max:
                feature_min = float(np.nanmin(feature_value))
                feature_max = float(np.nanmax(feature_value))
            if feature_max > feature_min:
                scaled_feature[finite_feature] = (
                    feature_value[finite_feature] - feature_min
                ) / (feature_max - feature_min)
                scaled_feature[finite_feature] = np.clip(
                    scaled_feature[finite_feature],
                    0,
                    1
                )
            else:
                scaled_feature[finite_feature] = 0.5

        shap_min = float(np.nanmin(shap_x))
        shap_max = float(np.nanmax(shap_x))
        quantized_x = np.round(
            n_bins * (shap_x - shap_min) / (shap_max - shap_min + 1e-8)
        )
        density_order = np.argsort(
            quantized_x + rng.standard_normal(n_samples) * 1e-6
        )
        layer = 0
        last_bin = -1
        jitter = np.zeros(n_samples, dtype=float)
        for sample_idx in density_order:
            if quantized_x[sample_idx] != last_bin:
                layer = 0
            jitter[sample_idx] = (
                np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            )
            layer += 1
            last_bin = quantized_x[sample_idx]
        jitter *= 0.9 * (row_height / np.max(jitter + 1))

        frames.append(pd.DataFrame({
            "sample_id": sample_ids[plot_order],
            "algorithm": sample_meta["algorithm"].to_numpy()[plot_order],
            "year_pair": sample_meta["year_pair"].to_numpy()[plot_order],
            "feature": feature,
            "feature_rank": int(feature_rank[feature_idx]),
            "feature_y": float(feature_center),
            "beeswarm_y": feature_center + jitter,
            "normalized_shap_value": shap_x,
            "feature_value": feature_value,
            "feature_value_scaled_0_1": scaled_feature,
            "relative_contribution_percent": float(
                global_relative[feature_idx] * 100
            )
        }))

    return pd.concat(frames, ignore_index=True)


def _write_shap_source_workbook(
    global_importance,
    model_year_importance,
    beeswarm_source,
    output_path
):
    '''
    保存SHAP柱状图和蜂群图展示样本的Excel源数据，并设置基础表格格式。

    :param global_importance: 总体变量贡献及排序表
    :param model_year_importance: 28个算法年份组合的变量贡献明细
    :param beeswarm_source: 蜂群图逐点源数据
    :param output_path: Excel文件输出路径
    :return: None
    '''
    from openpyxl.styles import Alignment, Font, PatternFill

    n_feature_rows = int(beeswarm_source["feature"].nunique())
    beeswarm_settings = pd.DataFrame({
        "setting": [
            "x_column",
            "y_column",
            "color_column",
            "label_center_column",
            "display_samples",
            "stratification",
            "x_bins",
            "row_height",
            "y_axis_min",
            "y_axis_max",
            "y_tick_centers",
            "recommended_bar_height",
            "matplotlib_marker_area_pt2",
            "recommended_origin_symbol_size_pt",
            "point_alpha",
            "origin_symbol_transparency_percent",
            "color_scale"
        ],
        "value": [
            "normalized_shap_value",
            "beeswarm_y",
            "feature_value_scaled_0_1",
            "feature_y",
            MAX_BEESWARM_DISPLAY_SAMPLES,
            "algorithm + year_pair",
            100,
            0.4,
            0.5,
            n_feature_rows + 0.5,
            f"1, 2, ..., {n_feature_rows}",
            0.6,
            16,
            "3–4",
            0.65,
            35,
            "blue-white-red; Low=0, High=1"
        ]
    })
    feature_axis = (
        beeswarm_source[[
            "feature_y",
            "feature_rank",
            "feature",
            "relative_contribution_percent"
        ]]
        .drop_duplicates()
        .sort_values("feature_y")
        .reset_index(drop=True)
    )
    sheets = {
        "Global_Importance": global_importance,
        "ModelYear_Importance": model_year_importance,
        "Feature_Axis": feature_axis,
        "Beeswarm_Data": beeswarm_source,
        "Beeswarm_Settings": beeswarm_settings
    }
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            worksheet.sheet_view.showGridLines = False

            for cell in worksheet[1]:
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.font = Font(color="FFFFFF", bold=True, name="Arial")
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # 只按表头和前200行估计列宽，避免无意义地遍历全部单元格。
            for column_idx, column_name in enumerate(table.columns, start=1):
                sample_values = table.iloc[:200, column_idx - 1].astype(str)
                max_length = max(
                    [len(str(column_name)), *sample_values.map(len).tolist()]
                )
                width = min(max(max_length + 2, 11), 30)
                column_letter = worksheet.cell(1, column_idx).column_letter
                worksheet.column_dimensions[column_letter].width = width


def export_global_shap_source_data():
    '''
    汇总并导出四分类器和七个年份对的OOF SHAP源数据。
    每个分类器和年份对先转为相对贡献，再对28组结果等权平均。相对贡献使用
    完整汇总结果，蜂群图源数据分层抽取固定数量的样本并采用SHAP原生堆叠。

    :return: None
    '''
    set_plot_style()
    all_values = []
    all_x = []
    all_meta = []
    relative_rows = []
    source_rows = []
    feature_names = None

    log("开始计算四分类器的 OOF SHAP 综合结果...")
    for algo_name, prefix in CONFIG["MODELS_INFO"].items():
        for year_pair in CONFIG["YEAR_PAIRS"]:
            values, X_oof, relative = compute_shap_for_algo_year(
                algo_name,
                prefix,
                year_pair
            )
            if values is None or X_oof is None or relative is None:
                continue

            current_features = list(X_oof.columns)
            if feature_names is None:
                feature_names = current_features
            elif current_features != feature_names:
                raise RuntimeError(
                    f"{algo_name} {year_pair} 的 SHAP 特征顺序不一致。"
                )

            all_values.append(values)
            all_x.append(X_oof)
            all_meta.append(pd.DataFrame({
                "algorithm": np.repeat(algo_name, len(X_oof)),
                "year_pair": np.repeat(year_pair, len(X_oof))
            }))
            relative_rows.append(relative)

            for feature, contribution in zip(feature_names, relative):
                source_rows.append({
                    "algorithm": algo_name,
                    "year_pair": year_pair,
                    "feature": feature,
                    "relative_contribution_percent": float(contribution * 100)
                })

    if not all_values or not all_x or not relative_rows:
        log("[跳过] 没有可用SHAP结果，无法导出总SHAP源数据。")
        return

    relative_matrix = np.vstack(relative_rows)
    global_relative = np.mean(relative_matrix, axis=0)

    group_source = pd.DataFrame(source_rows)
    global_source = pd.DataFrame({
        "algorithm": "Model-averaged",
        "year_pair": "All",
        "feature": feature_names,
        "relative_contribution_percent": global_relative * 100
    })
    pd.concat(
        [group_source, global_source],
        ignore_index=True
    ).to_csv(
        os.path.join(
            ENSEMBLE_EVAL_DIR,
            "Global_OOF_SHAP_Relative_Contribution.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    combined_values = np.vstack(all_values)
    combined_x = pd.concat(all_x, axis=0, ignore_index=True)
    combined_meta = pd.concat(all_meta, axis=0, ignore_index=True)

    # 防止总样本量过大
    if len(combined_x) > MAX_TOTAL_SHAP_SAMPLES:
        idx = np.random.RandomState(CONFIG["RANDOM_STATE"]).choice(
            len(combined_x),
            MAX_TOTAL_SHAP_SAMPLES,
            replace=False
        )
        combined_values = combined_values[idx]
        combined_x = combined_x.iloc[idx].reset_index(drop=True)
        combined_meta = combined_meta.iloc[idx].reset_index(drop=True)

    contribution_column = "relative_contribution_percent"
    group_stats = (
        group_source.groupby("feature")[contribution_column]
        .agg(["median", "std", "min", "max", "count"])
        .reindex(feature_names)
    )
    rank = np.empty(len(feature_names), dtype=np.int16)
    rank[np.argsort(-global_relative)] = np.arange(1, len(feature_names) + 1)
    global_importance = pd.DataFrame({
        "rank": rank,
        "feature": feature_names,
        "relative_contribution_percent": global_relative * 100,
        "median_model_year_contribution_percent": group_stats["median"].to_numpy(),
        "sd_model_year_contribution_percent": group_stats["std"].to_numpy(),
        "min_model_year_contribution_percent": group_stats["min"].to_numpy(),
        "max_model_year_contribution_percent": group_stats["max"].to_numpy(),
        "n_model_years": group_stats["count"].astype(int).to_numpy()
    }).sort_values("rank").reset_index(drop=True)

    model_year_importance = group_source.merge(
        global_importance[["feature", "rank"]],
        on="feature",
        how="left",
        validate="many_to_one"
    )[[
        "algorithm",
        "year_pair",
        "rank",
        "feature",
        "relative_contribution_percent"
    ]].sort_values(["algorithm", "year_pair", "rank"]).reset_index(drop=True)

    display_values, display_x, display_meta = _sample_shap_display_rows(
        combined_values,
        combined_x,
        combined_meta
    )
    beeswarm_source = _build_shap_beeswarm_source(
        display_values,
        display_x,
        display_meta,
        global_relative
    )
    shap_workbook_path = os.path.join(
        ENSEMBLE_EVAL_DIR,
        "Global_OOF_SHAP_Source_Data.xlsx"
    )
    _write_shap_source_workbook(
        global_importance,
        model_year_importance,
        beeswarm_source,
        shap_workbook_path
    )

    log(
        "[完成] 四分类器OOF SHAP相对贡献表和Excel源数据已输出；"
        f"蜂群图表包含{len(display_x)}个分层抽样展示样本。"
    )

# =========================================================
# 六、总评估主函数
# =========================================================
def run_ensemble_global_evaluation():
    '''
    汇总并输出四分类器与等权集成模型的全局评价结果。
    输出合并年份后的ROC、PR、最终三分类混淆矩阵、特征相关图和OOF SHAP源数据。

    :return: None
    '''
    log("=" * 50)
    log("开始生成四模型与等权集成的总评估图...")
    log("=" * 50)

    # 1. 复原四模型 OOF 预测
    all_oof = collect_all_oof_predictions()

    # 2. 对完全对齐的四分类器 OOF 概率进行等权集成
    ensemble_df = build_equal_weight_ensemble_oof(all_oof)

    # 3. 最终混淆矩阵、ROC、PR及其Origin源数据
    if all_oof:
        export_global_evaluation_source_data(all_oof, ensemble_df)
        plot_global_roc_pr_curves(all_oof, ensemble_df)
        log("[完成] 四分类器及等权集成的总ROC、总PR图和源数据已输出。")

    if ensemble_df is not None and not ensemble_df.empty:
        plot_ensemble_cm(ensemble_df)
        log("[完成] 等权集成总混淆矩阵已输出。")

    # 4. 总特征相关图
    plot_total_feature_correlation()

    # 5. 总 SHAP 解释源数据
    export_global_shap_source_data()

    log(f"总评估结果全部输出完成，请查看：{ENSEMBLE_EVAL_DIR}")


# =========================================================
# 七、主程序入口
# =========================================================
if __name__ == "__main__":
    log("="*50)
    log("开始多模型等权集成反演...")
    log("=" * 50)

    # 1. 逐年份对输出等权集成 tif 和预览图
    for yp in CONFIG["YEAR_PAIRS"]:
        run_ensemble(yp)

    # 2. 输出总混淆矩阵、总ROC、总PR、总相关性、总SHAP
    run_ensemble_global_evaluation()

    log(f"\n任务全部完成！请在 {ENSEMBLE_OUT_DIR} 文件夹查看结果。")
