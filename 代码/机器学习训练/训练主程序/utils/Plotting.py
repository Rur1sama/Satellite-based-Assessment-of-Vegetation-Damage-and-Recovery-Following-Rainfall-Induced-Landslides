"""
==============================================================================
Project       : Slide_dect
Module Name   : Plotting.py
File Path     : 代码/机器学习训练/训练主程序/utils/Plotting.py
File Created  : 2026-05-28 22:08:57
Description   : 制图函数--结果输出模块 
Lines         : 181
==============================================================================
"""

#!/usr/bin/env python
# coding=utf-8
from utils.head import * # 导入所有头文件
from utils.__init__ import * 

def plot_feature_correlation(df_features, year_pair, out_dir):
    '''
    计算特征相关性图

    :param df_features: 输入特征DataFrame
    :param year_pair: 指定年份对；“字符串”
    :param out_dir: 输出路径
    '''
    corr = df_features.corr()
    annot_matrix = corr.copy().astype(str)
    for i, col_i in enumerate(df_features.columns):
        for j, col_j in enumerate(df_features.columns):
            if i != j:
                r, p = pearsonr(df_features[col_i], df_features[col_j])
                if p < 0.001: star = "***"
                elif p < 0.01: star = "**"
                elif p < 0.05: star = "*"
                else: star = ""
                annot_matrix.iloc[i, j] = star if i < j else f"{r:.2f}"
            else:
                annot_matrix.iloc[i, j] = "1.00"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=annot_matrix, fmt="", cmap='RdBu', center=0, square=True)
    plt.title(f"Feature Correlation - {year_pair}")
    plt.savefig(os.path.join(out_dir, f"Corr_Star_{year_pair}.png"), dpi=600, bbox_inches='tight')
    plt.close()

def plot_year_feature_correlation(df, year_pair, out_dir):
    '''
    绘制单个年份对的特征相关性矩阵。
    '''
    features_to_plot = df.drop(columns=['idx', 'block', 'label', 'curvature', 'aspect'], errors='ignore')
    plot_feature_correlation(features_to_plot, year_pair, out_dir)

def plot_fold_diagnostics(y_true, y_pred, y_prob, year_pair, fold_idx, CM_DIR, ROC_DIR, PR_DIR):
    '''
    绘制单折训练结果图，包括三分类混淆矩阵、滑坡/非滑坡检测矩阵、ROC 与 PR 曲线。
    '''
    cm_multi = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm_multi, display_labels=["Background", "Slide", "Disturb"])
    disp.plot(cmap='Blues')
    plt.title(f"Fold {fold_idx}")
    plt.savefig(os.path.join(CM_DIR, f"CM_{year_pair}_F{fold_idx}.png"))
    plt.close()

    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    total_f = len(y_true_bin)
    cm_4dim = np.array([[tp_f/total_f, fp_f/total_f], [fn_f/total_f, tn_f/total_f]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_4dim, annot=True, fmt='.2%', cmap='Greens', cbar=False,
                xticklabels=['Pred Slide', 'Pred Rest'], yticklabels=['True Slide', 'True Rest'])
    plt.title(f"Detection Stats {year_pair} F{fold_idx}")
    plt.savefig(os.path.join(CM_DIR, f"CM_4Dim_{year_pair}_F{fold_idx}.png"))
    plt.close()

    if len(np.unique(y_true_bin)) > 1:
        fpr_f, tpr_f, _ = roc_curve(y_true_bin, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr_f, tpr_f, label=f'AUC={auc(fpr_f, tpr_f):.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.savefig(os.path.join(ROC_DIR, f"ROC_{year_pair}_F{fold_idx}.png"))
        plt.close()

        prec_f, rec_f, _ = precision_recall_curve(y_true_bin, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(rec_f, prec_f)
        plt.savefig(os.path.join(PR_DIR, f"PR_{year_pair}_F{fold_idx}.png"))
        plt.close()

def plot_feature_importance(feature_names, fi_list, year_pair, FI_DIR):
    '''
    绘制多折平均特征重要性排序图。
    '''
    if not fi_list:
        return

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.mean(fi_list, axis=0)
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', hue='Feature', legend=False)
    plt.savefig(os.path.join(FI_DIR, f"FI_{year_pair}.png"))
    plt.close()

def create_summary_grid(image_dir, prefix, out_filename, cols=3):
    '''
    将指定路径的所有图片制作成汇总图

    :param image_dir: 图片路径
    :param prefix: 前缀
    :param out_filename: 输出名称
    :param cols: 汇总图列数, defaults to 3
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

# =========================================================
# 全局评估指标计算函数
# =========================================================
def calculate_global_metrics(all_true, all_prob, all_pred, all_train_dfs, GLOBAL_DIR):
    '''
    计算全局汇总参数。包括混淆矩阵（4格加9格）、ROC曲线、PR曲线、特征相关图

    :param all_true: 全部样本标签  （0=背景、1=滑坡、2=扰动）
    :param all_prob: 全部预测的概率 
    :param all_pred: 全部预测的结果（0=背景、1=滑坡、2=扰动） 
    :param all_train_dfs: 全部的训练数据DataFrame
    :param GLOBAL_DIR: 全局汇总的输出路径
    '''
    all_true, all_prob, all_pred = np.array(all_true), np.array(all_prob), np.array(all_pred)
    all_true_bin = (all_true == 1).astype(int)
    all_pred_bin = (all_pred == 1).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(all_true_bin, all_pred_bin).ravel()
    total = len(all_true_bin)
    
    log("\n【全局汇总结果】")
    log(f"\n正确的  (TP%) : {tp/total:.2%}\n错判了 (FP%) : {fp/total:.2%}\n漏判了 (FN%) : {fn/total:.2%}\n正确背景(TN%) : {tn/total:.2%}")
    log(f"正确率 (Acc) : {(tp+tn)/total:.2%}")

    cm_perc = confusion_matrix(all_true, all_pred).astype('float') / len(all_true)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues',  
                xticklabels=['Background', 'Slide', 'Disturb'], yticklabels=['Background', 'Slide', 'Disturb'])
    plt.savefig(os.path.join(GLOBAL_DIR, "Global_CM_9DPercentage.png"), dpi=600); plt.close()
    
    cm_4dim_global = np.array([[tp/total, fp/total], [fn/total, tn/total]])
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_4dim_global, annot=True, fmt='.2%', cmap='Blues', annot_kws={"size": 16},
                xticklabels=['Predicted Landslide', 'Predicted Background'], 
                yticklabels=['Actual Landslide', 'Actual Background'])
    plt.savefig(os.path.join(GLOBAL_DIR, "Global_CM_4Dimensions.png"), dpi=600); plt.close()
    
    fpr, tpr, _ = roc_curve(all_true_bin, all_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, lw=2, label=f'Global ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.savefig(os.path.join(GLOBAL_DIR, "Global_ROC.png"), dpi=600, bbox_inches='tight'); plt.close()
    
    precision_vals, recall_vals, _ = precision_recall_curve(all_true_bin, all_prob)
    plt.figure(figsize=(8, 6)); plt.plot(recall_vals, precision_vals, lw=2, label='Global PR')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.savefig(os.path.join(GLOBAL_DIR, "Global_PR.png"), dpi=600, bbox_inches='tight'); plt.close()
    
    global_train_df = pd.concat(all_train_dfs, ignore_index=True)
    features_to_plot = global_train_df.drop(columns=['label', 'idx', 'block', 'curvature', 'aspect'], errors='ignore')
    plot_feature_correlation(features_to_plot, "Global_Average", GLOBAL_DIR)