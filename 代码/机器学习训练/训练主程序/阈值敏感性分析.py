"""
==============================================================================
Project       : Slide_dect
Module Name   : 阈值敏感性分析.py
Description   : 基于已有OOF预测和年度集成概率图开展滑坡阈值敏感性分析
==============================================================================
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENSEMBLE_DIR = PROJECT_ROOT / "Output" / "LandslideProducts" / "Data" / "annual_probability_maps"
EVALUATION_DIR = PROJECT_ROOT / "Output" / "TrainingResults" / "Ensemble_Evaluation"
STUDY_AREA_PATH = PROJECT_ROOT / "Input" / "Training" / "ExampleData" / "StudyArea" / "study_area_boundary.shp"

YEAR_PAIRS = ("18-19", "19-20", "20-21", "21-22", "22-23", "23-24", "24-25")
OOF_FILES = {
    "lightGBM": EVALUATION_DIR / "OOF_lightGBM.csv",
    "XGBoost": EVALUATION_DIR / "OOF_XGBoost.csv",
    "CatBoost": EVALUATION_DIR / "OOF_CatBoost.csv",
    "Random Forest": EVALUATION_DIR / "OOF_Random_Forest.csv",
}

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
PRIMARY_THRESHOLD = 0.8
CONNECTIVITY = 8
MIN_PATCH_AREA_M2 = 100.0


def _safe_ratio(numerator, denominator):
    '''
    安全计算比值，避免评价指标分母为零。

    :param numerator: 比值分子
    :param denominator: 比值分母
    :return: 浮点比值；分母为零时返回 NaN
    '''
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _event_year_from_year_pair(year_pair):
    '''
    从年份对提取后一景影像对应的名义滑坡事件年份。

    :param year_pair: Sentinel-2 年份对，例如 18-19
    :return: 四位数事件年份
    '''
    later_year = str(year_pair).split("-")[-1]
    if len(later_year) == 2:
        return 2000 + int(later_year)
    return int(later_year)


def load_equal_weight_oof_probability():
    '''
    读取四个分类器已有的OOF结果，并逐样本计算等权平均滑坡概率。
    四张表必须共享完全一致的年份、样本、标签、空间块和折编号。

    :return: 真实二分类标签和等权集成滑坡概率
    '''
    key_columns = ["year_pair", "idx", "label", "block", "fold_id"]
    use_columns = [*key_columns, "prob_1"]
    base_keys = None
    y_true = None
    probability_sum = None

    for model_name, oof_path in OOF_FILES.items():
        if not oof_path.exists():
            raise FileNotFoundError(f"找不到{model_name}的OOF结果: {oof_path}")

        table = pd.read_csv(oof_path, usecols=use_columns)
        table = table.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
        current_keys = table[key_columns]

        if base_keys is None:
            base_keys = current_keys.copy()
            y_true = (table["label"].to_numpy(dtype=int) == 1).astype(np.uint8)
            probability_sum = table["prob_1"].to_numpy(dtype=np.float64)
        else:
            if len(current_keys) != len(base_keys) or not current_keys.equals(base_keys):
                raise RuntimeError(
                    f"{model_name}的OOF样本与其他分类器不完全一致，不能等权集成。"
                )
            probability_sum += table["prob_1"].to_numpy(dtype=np.float64)

        print(f"已读取 {model_name}: {len(table):,} 个OOF样本")

    ensemble_probability = probability_sum / float(len(OOF_FILES))
    return y_true, ensemble_probability


def calculate_threshold_metrics(y_true, y_probability, threshold):
    '''
    计算指定阈值下等权集成OOF预测的二分类评价指标。
    滑坡为正类，背景和其他扰动共同构成非滑坡类。

    :param y_true: OOF样本真实二分类标签
    :param y_probability: OOF样本等权集成滑坡概率
    :param threshold: 滑坡概率判定阈值
    :return: 混淆矩阵计数及阈值相关评价指标
    '''
    y_predicted = (y_probability >= float(threshold)).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1]
    ).ravel()

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    f1 = (
        _safe_ratio(2.0 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall)
        else np.nan
    )
    balanced_accuracy = (
        float(np.mean([recall, specificity]))
        if np.isfinite(recall) and np.isfinite(specificity)
        else np.nan
    )

    return {
        "oof_n_samples": int(len(y_true)),
        "oof_n_observed_landslide": int(y_true.sum()),
        "oof_n_predicted_landslide": int(y_predicted.sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "accuracy_percent": _safe_ratio(tp + tn, len(y_true)) * 100.0,
        "precision_percent": precision * 100.0,
        "recall_percent": recall * 100.0,
        "specificity_percent": specificity * 100.0,
        "balanced_accuracy_percent": balanced_accuracy * 100.0,
        "f1_percent": f1 * 100.0,
    }


def count_landslide_patches(probability_path, threshold, study_area):
    '''
    按指定阈值从年度概率图构建滑坡斑块并统计数量。
    统计规则与生态服务分析保持一致：江湾镇边界裁剪、8邻域连通和
    100 m2最小斑块面积。

    :param probability_path: 年度等权集成概率栅格路径
    :param threshold: 滑坡概率判定阈值
    :param study_area: 江湾镇边界 GeoDataFrame
    :return: 候选连通斑块数、保留滑坡数及其面积
    '''
    with rasterio.open(probability_path) as src:
        if src.crs is None:
            raise ValueError(f"概率栅格缺少CRS: {probability_path}")
        if src.crs.is_geographic:
            raise ValueError(
                f"概率栅格必须使用米制投影坐标系: {probability_path}"
            )

        probability = src.read(1, masked=True)
        probability_values = np.ma.filled(probability, np.nan)
        positive_mask = (
            ~np.ma.getmaskarray(probability)
            & np.isfinite(probability_values)
            & (probability_values >= float(threshold))
        )
        positive_pixel_count = int(positive_mask.sum())
        if positive_pixel_count == 0:
            return {
                "threshold_positive_pixel_count": 0,
                "candidate_component_count": 0,
                "candidate_area_ha": 0.0,
                "predicted_landslide_count": 0,
                "retained_landslide_area_ha": 0.0,
            }

        projected_study_area = study_area.to_crs(src.crs)
        if hasattr(projected_study_area.geometry, "union_all"):
            study_union = projected_study_area.geometry.union_all()
        else:
            study_union = projected_study_area.geometry.unary_union

        candidate_count = 0
        candidate_area_m2 = 0.0
        retained_count = 0
        retained_area_m2 = 0.0

        for geometry_mapping, value in shapes(
            positive_mask.astype("uint8"),
            mask=positive_mask,
            transform=src.transform,
            connectivity=CONNECTIVITY,
        ):
            if int(value) != 1:
                continue

            geometry = shape(geometry_mapping)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            geometry = geometry.intersection(study_union)
            if geometry.is_empty:
                continue

            area_m2 = float(geometry.area)
            candidate_count += 1
            candidate_area_m2 += area_m2
            if area_m2 >= MIN_PATCH_AREA_M2:
                retained_count += 1
                retained_area_m2 += area_m2

    return {
        "threshold_positive_pixel_count": positive_pixel_count,
        "candidate_component_count": int(candidate_count),
        "candidate_area_ha": candidate_area_m2 / 10000.0,
        "predicted_landslide_count": int(retained_count),
        "retained_landslide_area_ha": retained_area_m2 / 10000.0,
    }


def write_excel_workbook(summary, annual_counts, settings, output_path):
    '''
    将阈值敏感性总表、逐年滑坡数和分析设置写入格式化Excel。

    :param summary: 阈值敏感性汇总表
    :param annual_counts: 各阈值和年份对应的滑坡统计表
    :param settings: 分析口径说明表
    :param output_path: Excel输出路径
    :return: None
    '''
    from openpyxl.styles import Alignment, Font, PatternFill

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        tables = {
            "Threshold_Summary": summary,
            "Annual_Counts": annual_counts,
            "Settings": settings,
        }
        for sheet_name, table in tables.items():
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
                column_letter = column_cells[0].column_letter
                worksheet.column_dimensions[column_letter].width = min(
                    max(len(header) + 2, observed_width + 2, 12),
                    36
                )


def run_threshold_sensitivity_analysis():
    '''
    执行0.5–0.9滑坡概率阈值敏感性分析并输出表格。
    精度指标来自七个年份对汇总的等权集成OOF预测；滑坡数来自七张年度
    等权概率图，二者共享相同阈值但具有不同统计单位。

    :return: 阈值敏感性分析Excel文件路径
    '''
    if not STUDY_AREA_PATH.exists():
        raise FileNotFoundError(f"找不到江湾镇边界: {STUDY_AREA_PATH}")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    study_area = gpd.read_file(STUDY_AREA_PATH)
    if study_area.empty or study_area.crs is None:
        raise ValueError("江湾镇边界为空或缺少CRS。")

    y_true, y_probability = load_equal_weight_oof_probability()
    summary_rows = []
    annual_rows = []

    for threshold in THRESHOLDS:
        print(f"正在分析阈值 {threshold:.1f} ...")
        threshold_annual_rows = []

        for year_pair in YEAR_PAIRS:
            probability_path = ENSEMBLE_DIR / f"landslide_probability_{year_pair.replace('-', '_')}.tif"
            if not probability_path.exists():
                raise FileNotFoundError(f"找不到年度概率图: {probability_path}")

            count_result = count_landslide_patches(
                probability_path,
                threshold,
                study_area
            )
            annual_row = {
                "threshold": float(threshold),
                "year_pair": year_pair,
                "event_year": _event_year_from_year_pair(year_pair),
                **count_result,
            }
            annual_rows.append(annual_row)
            threshold_annual_rows.append(annual_row)

        summary_rows.append({
            "threshold": float(threshold),
            "is_primary_threshold": bool(
                np.isclose(threshold, PRIMARY_THRESHOLD)
            ),
            **calculate_threshold_metrics(y_true, y_probability, threshold),
            "mapped_positive_pixel_count": int(sum(
                row["threshold_positive_pixel_count"]
                for row in threshold_annual_rows
            )),
            "candidate_component_count": int(sum(
                row["candidate_component_count"]
                for row in threshold_annual_rows
            )),
            "candidate_area_ha": float(sum(
                row["candidate_area_ha"]
                for row in threshold_annual_rows
            )),
            "predicted_landslide_count": int(sum(
                row["predicted_landslide_count"]
                for row in threshold_annual_rows
            )),
            "retained_landslide_area_ha": float(sum(
                row["retained_landslide_area_ha"]
                for row in threshold_annual_rows
            )),
        })

    summary = pd.DataFrame(summary_rows)
    annual_counts = pd.DataFrame(annual_rows)
    reference = summary[np.isclose(summary["threshold"], PRIMARY_THRESHOLD)]
    if not reference.empty:
        reference_count = int(reference.iloc[0]["predicted_landslide_count"])
        summary["count_difference_from_0_8"] = (
            summary["predicted_landslide_count"] - reference_count
        )
        summary["count_change_from_0_8_percent"] = (
            summary["count_difference_from_0_8"]
            / max(reference_count, 1)
            * 100.0
        )

    settings = pd.DataFrame({
        "item": [
            "thresholds",
            "primary_threshold",
            "accuracy_source",
            "positive_class",
            "probability_map_source",
            "connectivity",
            "minimum_patch_area_m2",
            "study_area",
            "interpretation_note",
        ],
        "value": [
            ", ".join(f"{value:.1f}" for value in THRESHOLDS),
            PRIMARY_THRESHOLD,
            "Pooled equal-weight ensemble OOF predictions across seven year pairs",
            "Landslide (label = 1); background and disturbance are non-landslide",
            "Seven annual equal-weight ensemble probability rasters",
            CONNECTIVITY,
            MIN_PATCH_AREA_M2,
            str(STUDY_AREA_PATH),
            "OOF metrics evaluate labelled pixels; predicted landslide counts are retained connected patches and are not sample counts.",
        ]
    })

    excel_path = EVALUATION_DIR / "Threshold_Sensitivity_Analysis.xlsx"
    csv_path = EVALUATION_DIR / "Threshold_Sensitivity_Summary.csv"
    write_excel_workbook(summary, annual_counts, settings, excel_path)
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n阈值敏感性分析汇总：")
    print(summary.to_string(index=False))
    print(f"\nExcel: {excel_path}")
    print(f"CSV:   {csv_path}")
    return excel_path


if __name__ == "__main__":
    run_threshold_sensitivity_analysis()
