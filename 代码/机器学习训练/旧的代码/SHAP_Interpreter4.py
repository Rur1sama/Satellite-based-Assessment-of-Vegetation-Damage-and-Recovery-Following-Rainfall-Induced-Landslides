"""
==============================================================================
Project       : Slide_dect
Module Name   : SHAP_Interpreter4.py
File Path     : 代码/机器学习训练/SHAP_Interpreter4.py
File Created  : 2026-05-14 10:41:39
Author        : leeyukhang
Description   : catboost shap
==============================================================================
"""

import os
from pathlib import Path
import glob
import gc
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

from catboost import Pool
# =========================================================
# 路径配置
# =========================================================
BASE_DIR = str(Path(__file__).resolve().parents[3] / "Output" / "TrainingResults" / "Legacy" / "CatBoost")
MODEL_DIR = os.path.join(BASE_DIR, "Saved_Models")
DATA_DIR = os.path.join(BASE_DIR, "Training_Data_CSVs")
SHAP_OUT_DIR = os.path.join(BASE_DIR, "SHAP_Plots")
GLOBAL_DIR = os.path.join(BASE_DIR, "global_summary")

os.makedirs(SHAP_OUT_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

# =========================================================
# 参数配置
# =========================================================

# 强烈建议先从小样本测试
MAX_SAMPLE_SIZE = 1000
# 滑坡类别 index
TARGET_CLASS_IDX = 1
# 是否生成 summary 图
SAVE_SUMMARY_PLOT = True
# 是否生成 bar 图
SAVE_BAR_PLOT = True


# =========================================================
# CatBoost 原生 SHAP 计算
# =========================================================
def compute_single_fold_shap(model_path, X_df, target_class_idx=1):
    """
    使用 CatBoost 原生 SHAP，避免 shap.TreeExplainer 崩溃
    """
    model = joblib.load(model_path)
    pool = Pool(X_df)
    shap_raw = model.get_feature_importance(
        data=pool,
        type="ShapValues",
        thread_count=-1,
        verbose=False
    )
    shap_raw = np.asarray(shap_raw)
    print(f"\nSHAP shape: {shap_raw.shape}")
    n_feat = X_df.shape[1]

    # =====================================================
    # 二分类 / 回归
    # shape: (n_samples, n_features + 1)
    # =====================================================
    if shap_raw.ndim == 2:
        values = shap_raw[:, :-1]
        base_values = shap_raw[:, -1]

    # =====================================================
    # 多分类
    # =====================================================
    elif shap_raw.ndim == 3:

        # 情况1:
        # (n_samples, n_classes, n_features + 1)
        if shap_raw.shape[2] == n_feat + 1:
            values = shap_raw[:, target_class_idx, :-1]
            base_values = shap_raw[:, target_class_idx, -1]

        # 情况2:
        # (n_samples, n_features + 1, n_classes)
        elif shap_raw.shape[1] == n_feat + 1:
            values = shap_raw[:, :-1, target_class_idx]
            base_values = shap_raw[:, -1, target_class_idx]

        else:
            raise ValueError(
                f"无法识别的多分类 SHAP shape: {shap_raw.shape}"
            )

    else:
        raise ValueError(
            f"Unexpected SHAP dimension: {shap_raw.shape}"
        )
    # 转 float32 节省内存
    values = values.astype(np.float32)
    base_values = base_values.astype(np.float32)
    # 主动释放
    del model
    del pool
    del shap_raw
    gc.collect()

    return values, base_values


# =========================================================
# 绘图
# =========================================================
def plot_and_save(explanation, X_sample, title_tag, out_dir):
    # -----------------------------------------------------
    # Summary Plot
    # -----------------------------------------------------
    if SAVE_SUMMARY_PLOT:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            explanation.values,
            X_sample,
            feature_names=explanation.feature_names,
            show=False,
            alpha=0.6, 
            plot_size=(10, 8)
        )
        plt.title(f"SHAP Summary - {title_tag}")
        plt.tight_layout()
        save_path = os.path.join(
            out_dir,
            f"SHAP_Summary_{title_tag}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"已保存: {save_path}")

    # -----------------------------------------------------
    # Bar Plot
    # -----------------------------------------------------
    if SAVE_BAR_PLOT:
        plt.figure(figsize=(10, 8))
        shap.plots.bar(
            explanation,
            show=False
        )
        plt.title(f"Feature Importance - {title_tag}")
        plt.tight_layout()
        save_path = os.path.join(
            out_dir,
            f"SHAP_Bar_{title_tag}.png"
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"已保存: {save_path}")


# =========================================================
# 单年份分析
# =========================================================
def run_shap_analysis(year_pair):

    print("\n===================================================")
    print(f"开始分析: {year_pair}")
    print("===================================================")

    # -----------------------------------------------------
    # 数据文件
    # -----------------------------------------------------
    data_path = os.path.join(
        DATA_DIR,
        f"train_{year_pair}.csv"
    )
    if not os.path.exists(data_path):
        print(f"找不到数据文件: {data_path}")
        return None, None, None
    # -----------------------------------------------------
    # 模型文件
    # -----------------------------------------------------
    model_pattern = os.path.join(
        MODEL_DIR,
        f"CB_{year_pair}_Fold*.pkl"
    )
    model_files = sorted(glob.glob(model_pattern))
    if len(model_files) == 0:
        print(f"找不到模型文件: {model_pattern}")
        return None, None, None
    print(f"发现 {len(model_files)} 个 fold 模型")

    # -----------------------------------------------------
    # 读取数据
    # -----------------------------------------------------
    df = pd.read_csv(data_path)
    X = df.drop(
        columns=[
            'label',
            'idx',
            'block',
            'curvature',
            'aspect'
        ],
        errors='ignore'
    )

    # -----------------------------------------------------
    # 抽样
    # -----------------------------------------------------
    sample_size = min(MAX_SAMPLE_SIZE, len(X))
    X_sample = X.sample(
        sample_size,
        random_state=42
    ).reset_index(drop=True)
    print(f"SHAP 样本数: {len(X_sample)}")

    # -----------------------------------------------------
    # 边算边平均
    # 避免 all_values 导致内存爆炸
    # -----------------------------------------------------
    sum_values = None
    sum_base_values = None
    n_fold = 0
    for model_path in tqdm(model_files, desc=f"{year_pair}"):
        try:
            val, base = compute_single_fold_shap(
                model_path,
                X_sample,
                TARGET_CLASS_IDX
            )
            if sum_values is None:

                sum_values = np.zeros_like(
                    val,
                    dtype=np.float32
                )
                sum_base_values = np.zeros_like(
                    base,
                    dtype=np.float32
                )
            sum_values += val
            sum_base_values += base
            n_fold += 1
            # 释放
            del val
            del base
            gc.collect()
        except Exception as e:
            print(f"\nFold 失败: {model_path}")
            print(f"错误信息: {e}")
            continue

    # -----------------------------------------------------
    # 检查
    # -----------------------------------------------------
    if n_fold == 0:
        print("所有 fold 都失败")
        return None, None, None

    # -----------------------------------------------------
    # 平均
    # -----------------------------------------------------
    avg_values = sum_values / n_fold
    avg_base_values = sum_base_values / n_fold

    # -----------------------------------------------------
    # Explanation
    # -----------------------------------------------------
    explanation = shap.Explanation(
        values=avg_values,
        base_values=avg_base_values,
        data=X_sample.values,
        feature_names=list(X_sample.columns)
    )

    # -----------------------------------------------------
    # 保存图
    # -----------------------------------------------------
    plot_and_save(
        explanation,
        X_sample,
        year_pair,
        SHAP_OUT_DIR
    )
    print(f"\n{year_pair} 分析完成")
    return avg_values, avg_base_values, X_sample


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":

    years = [
        "18-19","19-20","20-21","21-22","22-23","23-24","24-25"
    ]
    global_shaps = []
    global_bases = []
    global_xs = []
    for yp in years:
        values, bases, xs = run_shap_analysis(yp)
        if values is not None:
            sample_n = min(2000, len(xs))
        sample_idx = xs.sample(
            sample_n,
            random_state=42
        ).index
        sample_idx = np.sort(sample_idx)
        global_shaps.append(values[sample_idx])
        global_bases.append(bases[sample_idx])
        global_xs.append(xs.loc[sample_idx])
    # =====================================================
    # 全局汇总
    # =====================================================
    if len(global_shaps) > 0:
        print("\n===================================================")
        print("开始生成全局 SHAP")
        print("===================================================")
        combined_values = np.vstack(global_shaps)
        combined_bases = np.concatenate(global_bases)
        combined_x = pd.concat(
            global_xs,
            axis=0
        ).reset_index(drop=True)
        # ================= 控制全局图的样本总量 =================
        # 强制把最终画图的总点数控制在 5000 以内，避免图表“糊成一团”
        MAX_GLOBAL_SAMPLES = 2000
        if len(combined_x) > MAX_GLOBAL_SAMPLES:
            np.random.seed(42) # 保证每次生成的图表一致
            idx = np.random.choice(len(combined_x), MAX_GLOBAL_SAMPLES, replace=False)
            combined_values = combined_values[idx]
            combined_bases = combined_bases[idx]
            combined_x = combined_x.iloc[idx].reset_index(drop=True)
        # ================================================================

        global_explanation = shap.Explanation(
            values=combined_values,
            base_values=combined_bases, # 传入合并后的基准值，解决 AttributeError
            data=combined_x.values,
            feature_names=list(combined_x.columns)
        )
        shap.dependence_plot("delta_ndvi", combined_values, combined_x, display_features=combined_x)
        plot_and_save(global_explanation,combined_x,"Global_Average",GLOBAL_DIR)
        print("\n全局汇总完成")

    print("\n所有 SHAP 分析结束！")
