"""
==============================================================================
Project       : Slide_dect
Module Name   : SHAP_Interpreter.py
File Path     : code/Model training and landslide mapping/Legacy scripts/SHAP_Interpreter.py
File Created  : 2026-05-14 10:36:14
Author        : leeyukhang
Description   : lightGBM shap解释
==============================================================================
"""

import os
from pathlib import Path
import glob
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# =========================
# 路径配置 (完全同步 训练1.py)
# =========================
BASE_DIR = str(Path(__file__).resolve().parents[3] / "Output" / "TrainingResults" / "Legacy" / "LightGBM")
MODEL_DIR = os.path.join(BASE_DIR, "Saved_Models")
DATA_DIR = os.path.join(BASE_DIR, "Training_Data_CSVs") # 训练1.py 中的 TRAIN_DATA_DIR
SHAP_OUT_DIR = os.path.join(BASE_DIR, "SHAP_Plots")
GLOBAL_DIR = os.path.join(BASE_DIR, "global_summary") # 训练1.py 中的 Global_DIR

# 自动创建目录
for d in [SHAP_OUT_DIR, GLOBAL_DIR]:
    os.makedirs(d, exist_ok=True)

def run_shap_analysis(year_pair):
    """
    针对单一年份进行多折模型平均的 SHAP 分析
    """
    # 1. 检查数据文件 (匹配 训练1.py 的 df_train.to_csv 逻辑)
    data_path = os.path.join(DATA_DIR, f"train_{year_pair}.csv") #
    if not os.path.exists(data_path):
        print(f"跳过：找不到训练数据 {data_path}")
        return None, None

    # 2. 检查模型文件 (匹配 训练1.py 的 joblib.dump 逻辑)
    model_pattern = os.path.join(MODEL_DIR, f"LGBM_{year_pair}_Fold*.pkl") #
    model_files = glob.glob(model_pattern)
    if not model_files:
        print(f"跳过：找不到模型文件 {model_pattern}")
        return None, None

    # 3. 数据处理
    df = pd.read_csv(data_path)
    # 确保剔除列与 训练1.py 完全一致: drop(columns=['label','idx','block'])
    X = df.drop(columns=['label', 'idx', 'block','curvature','aspect'], errors='ignore')
    # 采样 50000 个点进行解释
    X_sample = X.sample(min(2000, len(X)), random_state=42)

    all_values = []
    all_base_values = []
    
    print(f"正在分析 {year_pair} (共 {len(model_files)} 个 Folds)...")
    for m_path in tqdm(model_files, desc=f"SHAP {year_pair}", leave=False):
        model = joblib.load(m_path)
        explainer = shap.TreeExplainer(model)
        
        # 计算 SHAP 值 (check_additivity=False 防止轻微浮点误差报错)
        temp_explanation = explainer(X_sample, check_additivity=False)
        
        all_values.append(temp_explanation.values)
        all_base_values.append(temp_explanation.base_values)
        
        del model, explainer
        gc.collect()

    # 计算 10 折平均
    avg_values = np.mean(all_values, axis=0)
    avg_base_values = np.mean(all_base_values, axis=0)

    # 封装为 SHAP Explanation 对象
    # 多分类结果中，index 1 对应滑坡类
    target_class_idx = 1 
    # 提取滑坡类的 SHAP 值和基准值
    val_slice = avg_values[:, :, target_class_idx]
    base_slice = avg_base_values[:, target_class_idx]
    explanation = shap.Explanation(
        values=val_slice,
        base_values=base_slice,
        data=X_sample.values,
        feature_names=list(X.columns)
    )
    
    # 绘图保存 (年度图)
    plot_and_save(explanation, X_sample, year_pair, SHAP_OUT_DIR)
    
    # 返回 values, base_values 和 X数据，用于全局汇总
    return val_slice, base_slice, X_sample

def plot_and_save(explanation, X_sample, title_tag, out_dir):
    """
    通用绘图逻辑
    """
    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation, X_sample, show=False)
    plt.title(f"SHAP Summary - {title_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"SHAP_Summary_{title_tag}.png"), dpi=300)
    plt.close()

    # Bar Plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(explanation, show=False)
    plt.title(f"Feature Importance - {title_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"SHAP_Bar_{title_tag}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    years = ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"] #
    
    global_shaps = []
    global_bases = [] # 存储基准值
    global_xs = []

    for yp in years:
        res_values, res_bases, res_x = run_shap_analysis(yp)
        if res_values is not None:
            global_shaps.append(res_values)
            global_bases.append(res_bases) # 收集
            global_xs.append(res_x)
    
    # --- 全局汇总分析 ---
    if global_shaps:
        print("\n--- 正在生成全年份汇总 SHAP 分析 ---")
        # 合并所有年份的 SHAP 值和数据点
        combined_values = np.vstack(global_shaps)
        combined_bases = np.concatenate(global_bases) #  合并基准值
        combined_x = pd.concat(global_xs, axis=0)
        
        # 创建全局 Explanation
        # 构造包含 base_values 的全局 Explanation
        global_explanation = shap.Explanation(
            values=combined_values,
            base_values=combined_bases, # 传入合并后的基准值，解决 AttributeError
            data=combined_x.values,
            feature_names=list(combined_x.columns)
        )
        
        # 保存到 Global_DIR (global_summary 文件夹)
        plot_and_save(global_explanation, combined_x, "Global_Average", GLOBAL_DIR)
        print(f"全局汇总完成，存放在: {GLOBAL_DIR}")

    print("\n所有分析任务已完成！")
