"""
==============================================================================
Project       : Slide_dect
Module Name   : SHAP_Interpreter3.py
File Path     : code/Model training and landslide mapping/Legacy scripts/SHAP_Interpreter3.py
File Created  : 2026-05-14 10:40:43
Author        : leeyukhang
Description   : XGBoost shap解释
==============================================================================
"""

import os
from pathlib import Path
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# =========================
# 路径配置 (完全同步 训练2.py)
# =========================
BASE_DIR = str(Path(__file__).resolve().parents[3] / "Output" / "TrainingResults" / "Legacy" / "XGBoost")
MODEL_DIR = os.path.join(BASE_DIR, "Saved_Models")
DATA_DIR = os.path.join(BASE_DIR, "Training_Data_CSVs") # 训练2.py 中的 TRAIN_DATA_DIR
SHAP_OUT_DIR = os.path.join(BASE_DIR, "SHAP_Plots")
GLOBAL_DIR = os.path.join(BASE_DIR, "global_summary") # 训练2.py 中的 Global_DIR

# 自动创建目录
for d in [SHAP_OUT_DIR, GLOBAL_DIR]:
    os.makedirs(d, exist_ok=True)
def _compute_single_fold(m_path, X_data):
    """
    独立计算单个模型的 SHAP，用于并行化
    """
    import joblib
    import shap
    
    model = joblib.load(m_path)
    explainer = shap.TreeExplainer(model)
    
    # 【可选终极提速】如果依然觉得太慢，可以放弃 explainer() 对象调用
    # 改用近似计算：shap_vals = explainer.shap_values(X_data, approximate=True)
    
    temp_explanation = explainer(X_data, check_additivity=False)
    
    # 只返回 numpy array，避免对象跨进程传输带来的序列化开销
    return temp_explanation.values, temp_explanation.base_values

def run_shap_analysis(year_pair):
    """
    针对单一年份进行多折模型平均的 SHAP 分析 (优化版)
    """
    data_path = os.path.join(DATA_DIR, f"train_{year_pair}.csv")
    if not os.path.exists(data_path):
        print(f"跳过：找不到训练数据 {data_path}")
        return None, None

    model_pattern = os.path.join(MODEL_DIR, f"XGB_{year_pair}_Fold*.pkl")
    model_files = glob.glob(model_pattern)
    if not model_files:
        print(f"跳过：找不到模型文件 {model_pattern}")
        return None, None

    # 3. 数据处理
    df = pd.read_csv(data_path)
    # 确保剔除列与 训练2.py 的剔除列 完全一致: drop(columns=['label','idx','block'])
    X = df.drop(columns=['label', 'idx', 'block','curvature','aspect'], errors='ignore')
    
    # 采样 50000 个点，并【转换为 float32 提速】
    X_sample = X.sample(min(2000, len(X)), random_state=42).astype(np.float32)

    print(f"正在并行分析 {year_pair} (共 {len(model_files)} 个 Folds)...")
    
    # 【核心优化：使用 joblib 开启多进程并行】
    # n_jobs=-1 表示使用所有可用 CPU 核心
    results = Parallel(n_jobs=5, backend="loky")(
        delayed(_compute_single_fold)(m_path, X_sample) for m_path in model_files
    )

    # 拆解并行返回的结果
    all_values = [res[0] for res in results]
    all_base_values = [res[1] for res in results]

    # 计算平均
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
        
        # ================= 核心优化：控制全局图的样本总量 =================
        # 强制把最终画图的总点数控制在 5000 以内，避免图表“糊成一团”
        MAX_GLOBAL_SAMPLES = 5000
        if len(combined_x) > MAX_GLOBAL_SAMPLES:
            np.random.seed(42) # 保证每次生成的图表一致
            idx = np.random.choice(len(combined_x), MAX_GLOBAL_SAMPLES, replace=False)
            
            combined_values = combined_values[idx]
            combined_bases = combined_bases[idx]
            combined_x = combined_x.iloc[idx].reset_index(drop=True)
        # ================================================================

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
