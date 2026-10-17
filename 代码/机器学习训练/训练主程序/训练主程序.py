"""
==============================================================================
Project       : Slide_dect
Module Name   : 训练主程序.py
File Path     : 代码/机器学习训练/训练主程序/训练主程序.py
File Created  : 2026-05-20 14:12:19
Description   : 用于调度其他模块开始训练--主控模块
Lines         : 194
==============================================================================
"""

#!/usr/bin/env python
# coding=utf-8
from utils.head import * # 导入所有头文件
from utils.__init__ import * # 初始化参数配置、模型参数
from utils.base_utils import *
from utils.label_process import * 
from utils.Plotting import * # 绘图函数模块

# =========================================================
# 年度核心模型训练与交叉验证：训练逻辑与绘图逻辑分离
# =========================================================
MODEL_FEATURE_DROP_COLUMNS = [
    'label', 'idx', 'block', 'fold_id', 'curvature', 'aspect'
]

def train_groupkfold_models(year_pair, algo_name, prefix, X, y, groups, fold_ids, MODEL_SAVE_DIR, CM_DIR, ROC_DIR, PR_DIR):
    '''
    使用预先固定的空间折编号训练十个 GroupKFold 模型。
    每个折模型只预测对应验证样本，并保存模型供后续 OOF SHAP 分析。

    :param year_pair:       训练年份对
    :param algo_name:       算法名称 LightGBM、Random Forest、XGBoost、CatBoost
    :param prefix:          算法缩写前缀
    :param X:               训练特征
    :param y:               训练标签
    :param groups:          空间分组块
    :param fold_ids:        每个训练样本对应的固定折编号
    :param MODEL_SAVE_DIR:  当前算法模型保存路径
    :param CM_DIR:          混淆矩阵输出路径
    :param ROC_DIR:         ROC曲线输出路径
    :param PR_DIR:          PR曲线输出路径
    :return: 当前年份对的 OOF 真实标签、滑坡概率和预测类别
    '''
    year_y_true, year_y_prob, year_y_pred = [], [], []

    all_idx = np.arange(len(X))
    fold_ids_array = np.asarray(fold_ids, dtype=int)
    n_splits = CONFIG["N_SPLITS"]

    for fold_idx in tqdm(range(1, n_splits + 1),
        total=n_splits,
        desc=f"  [k折训练] {prefix}-{year_pair}",
        position=2,
        leave=False
    ):
        # 使用共享CSV中已经保存的fold编号划分训练集和验证集
        te = all_idx[fold_ids_array == fold_idx]
        tr = all_idx[fold_ids_array != fold_idx]

        if len(te) == 0:
            raise RuntimeError(
                f"{year_pair} Fold{fold_idx} 没有验证样本。"
            )

        # 检查同一个空间块是否同时出现在训练集和验证集
        train_blocks = set(groups.iloc[tr].tolist())
        test_blocks = set(groups.iloc[te].tolist())
        overlapping_blocks = train_blocks.intersection(test_blocks)

        if overlapping_blocks:
            raise RuntimeError(
                f"{year_pair} Fold{fold_idx} 出现空间块泄漏，"
                f"重叠空间块数量为 {len(overlapping_blocks)}。"
            )

        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]

        # 训练当前折模型
        model = get_model_instance(algo_name)
        model.fit(X_tr, y_tr)

        # 保存当前折模型
        model_filename = (
            f"{prefix}_{year_pair}_Fold{fold_idx}.pkl"
        )
        model_path = os.path.join(
            MODEL_SAVE_DIR,
            model_filename
        )
        joblib.dump(model, model_path)

        # 当前折只预测没有参与训练的验证样本
        preds = np.asarray(model.predict(X_te))

        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.flatten()

        # 找到“滑坡类1”在predict_proba中的对应列
        classes = np.asarray(model.classes_)
        landslide_columns = np.flatnonzero(classes == 1)

        if len(landslide_columns) != 1:
            raise RuntimeError(
                f"{prefix}-{year_pair}-Fold{fold_idx} "
                "无法确定滑坡类别的概率列。"
            )

        landslide_column = landslide_columns[0]
        y_prob = model.predict_proba(X_te)[
            :,
            landslide_column
        ]

        # 汇总十折OOF结果
        year_y_true.extend(y_te.tolist())
        year_y_pred.extend(preds.tolist())
        year_y_prob.extend(y_prob.tolist())

        # 绘制当前折诊断图
        plot_fold_diagnostics(
            y_te,
            preds,
            y_prob,
            year_pair,
            fold_idx,
            CM_DIR,
            ROC_DIR,
            PR_DIR
        )

        del model
        gc.collect()

    return year_y_true, year_y_prob, year_y_pred

def train_final_model(year_pair, algo_name, prefix, X, y, model_save_dir):
    '''
    使用当前年份对的全部训练样本重新训练一个最终制图模型。
    
        每个年份对最终保存：
        - 一个 Final LightGBM；
        - 一个 Final XGBoost；
        - 一个 Final CatBoost；
        - 一个 Final Random Forest。

    :param year_pair: 训练年份对
    :param algo_name: 算法名称 LightGBM、Random Forest、XGBoost、CatBoost
    :param prefix: 算法缩写前缀
    :param X: 训练特征
    :param y: 训练标签
    :param model_save_dir: 当前算法总模型保存路径
    :return: 该算法最终制图模型
    '''
    final_model = get_model_instance(algo_name)
    final_model.fit(X, y)

    model_path = os.path.join(
        model_save_dir,
        f"{prefix}_{year_pair}_Final.pkl"
    )
    joblib.dump(final_model, model_path)

    log(f"已保存全样本最终模型: {model_path}")
    return final_model

def save_final_prediction_raster(
    df,
    final_model,
    profile,
    h,
    w,
    current_output_dir,
    year_pair
):
    '''
    使用全部训练样本拟合的最终模型预测整张图。

    :param df: 当前年份对的全图像元特征表
    :param final_model: 使用全部共享训练样本拟合的最终分类器
    :param profile: 输出栅格使用的空间参考和元数据
    :param h: 输出栅格行数
    :param w: 输出栅格列数
    :param current_output_dir: 当前分类器的概率图输出目录
    :param year_pair: Sentinel-2 年份对
    :return: None
    '''
    X_full = df.drop(
        columns=MODEL_FEATURE_DROP_COLUMNS,
        errors="ignore"
    )

    classes = np.asarray(final_model.classes_)
    landslide_columns = np.flatnonzero(classes == 1)
    if len(landslide_columns) != 1:
        raise RuntimeError(
            f"{year_pair} 最终模型无法确定滑坡类别的概率列。"
        )

    final_prob = final_model.predict_proba(X_full)[
        :,
        landslide_columns[0]
    ]

    res_raster = np.full(h * w, np.nan)
    res_raster[df["idx"].values] = final_prob

    output_path = os.path.join(
        current_output_dir,
        f"Landslide_{year_pair}.tif"
    )

    log(f"正在保存 {output_path}")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(
            res_raster.reshape(h, w).astype(np.float32),
            1
        )

def run_model_training(year_pair, algo_name, prefix, CURRENT_OUTPUT_DIR, ROC_DIR, PR_DIR, CM_DIR, CORR_DIR, MODEL_SAVE_DIR, TRAIN_DATA_DIR, FI_DIR):
    '''
    运行全流程的单个年份对的模型训练、绘图与全图预测输出。
    十折模型仅用于折外验证；每种算法另用全部训练样本拟合最终模型并预测整图，
    四张最终概率图随后进行等权算术平均。
    程序固定折数计算

    :param year_pair:           当前预测年份对
    :param algo_name:           算法名称
    :param prefix:              算法缩写前缀
    :param CURRENT_OUTPUT_DIR:  目前导出路径
    :param ROC_DIR:             ROC曲线输出路径
    :param PR_DIR:              PR曲线输出路径
    :param CM_DIR:              混淆矩阵输出路径
    :param CORR_DIR:            相关性图输出路径
    :param MODEL_SAVE_DIR:      当前算法模型保存路径
    :param TRAIN_DATA_DIR:      抽样后的训练数据路径
    :param FI_DIR:              特征重要性保存路径
    :return: year_y_true        当前年份真实标签 
    \n       year_y_prob        当前年份预测概率
    \n       year_y_pred        当前年份预测标签
    \n       df_train           训练数据dataFrame（用于后续汇总绘图）
    '''
    df, profile, h, w = build_feature_dataframe(year_pair)
    plot_year_feature_correlation(df, year_pair, CORR_DIR)
    df_train = sample_training_dataframe(
        df,
        year_pair,
        TRAIN_DATA_DIR
    )
    X, y, groups, fold_ids = split_training_inputs(df_train)

    year_y_true, year_y_prob, year_y_pred = train_groupkfold_models(
        year_pair, algo_name, prefix, X, y, groups, fold_ids,
        MODEL_SAVE_DIR, CM_DIR, ROC_DIR, PR_DIR
    )
    # 十折交叉验证结束后，使用全部训练样本重新拟合最终模型
    final_model = train_final_model(
        year_pair=year_pair,
        algo_name=algo_name,
        prefix=prefix,
        X=X,
        y=y,
        model_save_dir=MODEL_SAVE_DIR
    )
    # 最终模型负责整图预测
    save_final_prediction_raster(
        df=df,
        final_model=final_model,
        profile=profile,
        h=h,
        w=w,
        current_output_dir=CURRENT_OUTPUT_DIR,
        year_pair=year_pair
    )

    return year_y_true, year_y_prob, year_y_pred, df_train

# =========================================================
# 5. 主程序入口：四模型循环训练控制
# =========================================================
if __name__ == "__main__":
    MODELS_INFO = CONFIG["MODELS_INFO"]
    def run_train():
        '''
        依次完成四个分类器和七个年份对的交叉验证、最终模型拟合与整图预测。

        :return: None
        '''
        # 外层大循环：依次训练每一个模型
        for algo_dir, prefix in tqdm(MODELS_INFO.items(), desc="[总体进度] 四大算法训练", position=0):
            tqdm.write(f"\n=======================================================")
            tqdm.write(f"[*] 开始启动机器学习算法训练流: {algo_dir} (前缀: {prefix})")
            tqdm.write(f"=======================================================")
            
            # 1. 直接动态定义清晰的标准路径变量（不使用字典）
            CURRENT_OUTPUT_DIR = os.path.join(CONFIG["MODELS_BASE_DIR"], algo_dir)
            
            ROC_DIR = os.path.join(CURRENT_OUTPUT_DIR, "ROC_curves")
            PR_DIR = os.path.join(CURRENT_OUTPUT_DIR, "PR_curves")
            CM_DIR = os.path.join(CURRENT_OUTPUT_DIR, "CM_folds")
            CORR_DIR = os.path.join(CURRENT_OUTPUT_DIR, "Correlation_plots")
            MODEL_SAVE_DIR = os.path.join(CURRENT_OUTPUT_DIR, "Saved_Models")
            TRAIN_DATA_DIR = CONFIG["SHARED_TRAIN_DIR"]
            FI_DIR = os.path.join(CURRENT_OUTPUT_DIR, "Feature_Importance")
            GLOBAL_DIR = os.path.join(CURRENT_OUTPUT_DIR, "global_summary")
            
            # 创建当前模型的所有对应子文件夹
            for d in [CURRENT_OUTPUT_DIR, ROC_DIR, PR_DIR, CM_DIR, CORR_DIR, MODEL_SAVE_DIR, TRAIN_DATA_DIR, FI_DIR, GLOBAL_DIR]:
            # for d in [CURRENT_OUTPUT_DIR, ROC_DIR, PR_DIR, CM_DIR, CORR_DIR, MODEL_SAVE_DIR, TRAIN_DATA_DIR, GLOBAL_DIR]:
                os.makedirs(d, exist_ok=True)
                
            g_true, g_prob, g_pred = [], [], []
            all_train_dfs = [] 
            
            # 2. 内层循环：处理所有的年份序列对
            for yp in tqdm(CONFIG["YEAR_PAIRS"], desc=f"[{algo_dir}] 年份处理进度", position=1, leave=False):
                log(f"正在处理 {algo_dir} 模型下的时序对: {yp} ...")
                try:
                    # 明明白白地把刚才创建好的路径变量通过传参送进函数中
                    y_t, y_p, y_pred, df_t = run_model_training(
                        yp, algo_dir, prefix, 
                        CURRENT_OUTPUT_DIR, ROC_DIR, PR_DIR, CM_DIR, CORR_DIR, MODEL_SAVE_DIR, TRAIN_DATA_DIR, FI_DIR
                        # CURRENT_OUTPUT_DIR, ROC_DIR, PR_DIR, CM_DIR, CORR_DIR, MODEL_SAVE_DIR, TRAIN_DATA_DIR
                    )
                    g_true.extend(y_t)
                    g_prob.extend(y_p)
                    g_pred.extend(y_pred)
                    all_train_dfs.append(df_t)
                except Exception as e:
                    log(f"[错误] 在处理年份对 {yp} 时发生致命异常: {e}")
                    continue
            
            # 3. 统计当前模型的全局评估指标并保存综合拼接图
            if g_true:
                calculate_global_metrics(g_true, g_prob, g_pred, all_train_dfs, GLOBAL_DIR)
                create_summary_grid(CORR_DIR, "Corr_Star_", "Summary_Correlation.png")
                # create_summary_grid(FI_DIR, "FI_", "Summary_FI.png")
                log(f"[✓] 算法 {algo_dir} 的所有训练与评估任务圆满完成！\n")        
        tqdm.write("\n 恭喜！全流程多算法综合模型训练全部结束！")
    run_train()
