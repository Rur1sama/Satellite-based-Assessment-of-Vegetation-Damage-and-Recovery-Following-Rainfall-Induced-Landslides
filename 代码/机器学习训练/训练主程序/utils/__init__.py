"""
==============================================================================
Project       : Slide_dect
Module Name   : __init__.py
File Path     : 代码/机器学习训练/综合训练版本/utils/__init__.py
File Created  : 2026-05-26 10:08:18
Description   : 初始化参数配置、模型参数
Lines         : 80
==============================================================================
"""

from utils.head import * # 导入所有头文件
# =========================================================
# 配置路径与参数
# =========================================================
CONFIG = {
    "OPTICAL_DIR": r"E:\python_base\Slide_dect\Input\光学栅格",
    "DEM_PATH": r"E:\python_base\Slide_dect\Input\高程数据\ALOS_PALSAR_DEM.tif",
    "GPM_DIR": r"E:\python_base\Slide_dect\Input\GPM_Time_Series",
    "MODELS_BASE_DIR": r"E:\python_base\Slide_dect\Output\ML_Results",
    "OUTPUT_DIR": r"E:\python_base\Slide_dect\Output",
    "LABEL_ROOT": r"E:\python_base\Slide_dect\Input\样本面手动标注结果",
    "BANDS": {"RED": 4, "NIR": 8, "Green":3},
    "YEAR_PAIRS": ["18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25"],
    # "YEAR_PAIRS": ["21-22","23-24"],
    # "YEAR_PAIRS": ["23-24"],
    # 定义四个模型所在的文件夹名称及对应的模型文件前缀
    "MODELS_INFO": {
        "lightGBM": "LGBM",
        "XGBoost": "XGB",
        "CatBoost": "CB",
        "Random Forest": "RF"
    },
    "RANDOM_STATE": 42,
    "N_SPLITS": 10
}
#----创建共享年度样本和固定fold----
CONFIG["SHARED_TRAIN_DIR"] = os.path.join(
    CONFIG["MODELS_BASE_DIR"],
    "Shared_Training_Data"
)
#----创建共享年度样本和固定fold----

#----舍弃部分不必要特征后训练----
MODEL_FEATURE_DROP_COLUMNS = ['label', 'idx', 'block', 'fold_id', 'curvature', 'aspect']
#----舍弃部分不必要特征后训练----

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

def get_model_instance(algo_name):
    '''
    模型初始化

    :param algo_name: 算法名称；字符串
    :raises ValueError: 未知算法名称：algo_name
    :return: 分类器instance
    '''
    if algo_name == "lightGBM":
        return lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=200, 
                                   learning_rate=0.05, force_col_wise=True, verbose=-1,
                                   predict_disable_shape_check=True,
                                   random_state=CONFIG["RANDOM_STATE"]) #LightGBM补上随机种子
    elif algo_name == "XGBoost":
        return xgb.XGBClassifier(objective='multi:softprob', num_class=3, n_estimators=200,
                             learning_rate=0.05, eval_metric='mlogloss', random_state=42, n_jobs=-1,
                             tree_method='hist', device="cuda")
    elif algo_name == "CatBoost":
        return CatBoostClassifier(loss_function='MultiClass', iterations=200, learning_rate=0.05,
                                  verbose=0, random_state=42, thread_count=-1)
    elif algo_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"未知算法名称: {algo_name}")
    

def get_feature_importance(model, algo_name):
    '''
    统一提取不同模型的特征重要性。
    '''
    if algo_name == "CatBoost":
        return model.get_feature_importance()
    return model.feature_importances_

def log(msg): 
    '''
    记录运行日志并打印输出
    '''
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    tqdm.write(line)
    with open(os.path.join(CONFIG["OUTPUT_DIR"], "run_log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return None
