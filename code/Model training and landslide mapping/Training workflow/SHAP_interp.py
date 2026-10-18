"""
==============================================================================
Project       : Slide_dect
Module Name   : SHAP_interp.py
File Path     : code/Model training and landslide mapping/Training workflow/SHAP_interp.py
File Created  : 2026-05-25 14:14:28
Description   : 统一的 SHAP 解释器，兼容 LightGBM, XGBoost, RF, CatBoost--结果输出模块
Lines         : 149
==============================================================================
"""
# 导入你现有的配置
from utils.__init__ import CONFIG, MODEL_FEATURE_DROP_COLUMNS, log
from catboost import Pool
from utils.head import *


# 全局 SHAP 抽样参数
MAX_SAMPLE_SIZE_PER_YEAR = 2000
MAX_GLOBAL_SAMPLES = 5000
TARGET_CLASS_IDX = 1  # 1 为滑坡类

def _compute_tree_shap_single_fold(m_path, X_data):
    """用于 RF 和 XGBoost 的并行 SHAP 计算"""
    model = joblib.load(m_path)
    explainer = shap.TreeExplainer(model)
    temp_explanation = explainer(X_data, check_additivity=False)
    
    vals = np.array(temp_explanation.values).copy() # 强制转换为 numpy 数组并执行深拷贝 (.copy())
    bases = np.array(temp_explanation.base_values).copy() # 彻底切断与 SHAP C++ 底层对象的内存映射关联，变成纯 Python 数据，让 Pickle 可以顺利序列化打包。
    del explainer, model, temp_explanation # 主动释放 explainer 对象，防止子进程内存泄漏
    return vals, bases

def compute_catboost_shap(model_path, X_df):
    """CatBoost 原生 SHAP 计算"""
    model = joblib.load(model_path)
    pool = Pool(X_df)
    shap_raw = np.asarray(model.get_feature_importance(data=pool, type="ShapValues", thread_count=-1, verbose=False))
    
    # 提取多分类中的滑坡类 (维度处理参考你的 Interpreter4)
    if shap_raw.shape[1] == X_df.shape[1] + 1:
        values = shap_raw[:, :-1, TARGET_CLASS_IDX]
        base_values = shap_raw[:, -1, TARGET_CLASS_IDX]
    else:
        values = shap_raw[:, TARGET_CLASS_IDX, :-1]
        base_values = shap_raw[:, TARGET_CLASS_IDX, -1]
        
    del model, pool, shap_raw
    gc.collect()
    return values.astype(np.float32), base_values.astype(np.float32)

def plot_and_save(explanation, X_sample, title_tag, out_dir):
    """统一保存 Summary 和 Bar 图"""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation.values, X_sample, feature_names=explanation.feature_names, show=False, alpha=0.6)
    plt.title(f"SHAP Summary - {title_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"SHAP_Summary_{title_tag}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.plots.bar(explanation, show=False)
    plt.title(f"Feature Importance - {title_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"SHAP_Bar_{title_tag}.png"), dpi=300)
    plt.close()

def run_shap_pipeline(algo_name, prefix):
    """执行单个算法的全套 SHAP 分析"""
    base_dir = os.path.join(CONFIG["MODELS_BASE_DIR"], algo_name)
    model_dir = os.path.join(base_dir, "Saved_Models")
    data_dir = os.path.join(base_dir, "Training_Data_CSVs")
    shap_out_dir = os.path.join(base_dir, "SHAP_Plots")
    global_dir = os.path.join(base_dir, "global_summary")
    
    os.makedirs(shap_out_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)

    global_shaps, global_bases, global_xs = [], [], []

    log(f"开始 {algo_name} 模型的 SHAP 分析...")
    
    for yp in CONFIG["YEAR_PAIRS"]:
        data_path = os.path.join(data_dir, f"train_{yp}.csv")
        model_pattern = os.path.join(model_dir, f"{prefix}_{yp}_Fold*.pkl")
        model_files = sorted(glob.glob(model_pattern))
        
        if not os.path.exists(data_path) or not model_files:
            continue

        df = pd.read_csv(data_path)
        X = df.drop(columns=MODEL_FEATURE_DROP_COLUMNS, errors='ignore')
        X_sample = X.sample(min(MAX_SAMPLE_SIZE_PER_YEAR, len(X)), random_state=42).astype(np.float32)

        # 根据算法选择计算策略
        if algo_name == "CatBoost":
            sum_val, sum_base = 0, 0
            for m_path in tqdm(model_files, desc=f"CatBoost {yp}"):
                val, base = compute_catboost_shap(m_path, X_sample)
                sum_val += val
                sum_base += base
            avg_values = sum_val / len(model_files)
            avg_base_values = sum_base / len(model_files)
            
        elif algo_name in ["Random Forest", "XGBoost"]:
            results = Parallel(n_jobs=-3, backend="threading")( # 最大cpu核心-2
                delayed(_compute_tree_shap_single_fold)(m_path, X_sample) for m_path in model_files
            )
            avg_values = np.mean([r[0] for r in results], axis=0)[:, :, TARGET_CLASS_IDX]
            avg_base_values = np.mean([r[1] for r in results], axis=0)[:, TARGET_CLASS_IDX]
            
        elif algo_name == "lightGBM":
            all_val, all_base = [], []
            for m_path in tqdm(model_files, desc=f"LGBM {yp}"):
                model = joblib.load(m_path)
                exp = shap.TreeExplainer(model)(X_sample, check_additivity=False)
                all_val.append(exp.values)
                all_base.append(exp.base_values)
            avg_values = np.mean(all_val, axis=0)[:, :, TARGET_CLASS_IDX]
            avg_base_values = np.mean(all_base, axis=0)[:, TARGET_CLASS_IDX]

        # 封装与绘图
        explanation = shap.Explanation(values=avg_values, base_values=avg_base_values, 
                                       data=X_sample.values, feature_names=list(X.columns))
        plot_and_save(explanation, X_sample, yp, shap_out_dir)
        
        global_shaps.append(avg_values)
        global_bases.append(avg_base_values)
        global_xs.append(X_sample)

    # ================= 全局汇总 =================
    if global_shaps:
        combined_values = np.vstack(global_shaps)
        combined_bases = np.concatenate(global_bases)
        combined_x = pd.concat(global_xs, axis=0).reset_index(drop=True)
        
        if len(combined_x) > MAX_GLOBAL_SAMPLES:
            idx = np.random.choice(len(combined_x), MAX_GLOBAL_SAMPLES, replace=False)
            combined_values, combined_bases = combined_values[idx], combined_bases[idx]
            combined_x = combined_x.iloc[idx].reset_index(drop=True)

        global_explanation = shap.Explanation(values=combined_values, base_values=combined_bases, 
                                              data=combined_x.values, feature_names=list(combined_x.columns))
        plot_and_save(global_explanation, combined_x, "Global_Average", global_dir)
        log(f"{algo_name} 全局汇总完成！")

if __name__ == "__main__":
    # 遍历所有配置的模型进行统一解释
    for algo_name, prefix in CONFIG["MODELS_INFO"].items():
        run_shap_pipeline(algo_name, prefix)
