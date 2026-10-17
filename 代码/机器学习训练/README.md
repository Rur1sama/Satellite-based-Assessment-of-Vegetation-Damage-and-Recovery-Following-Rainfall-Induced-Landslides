# Model training and landslide mapping | 模型训练与滑坡制图

This guide describes the primary training workflow in this repository. The older scripts in `旧的代码/` are retained for reference; use `训练主程序/` for the current workflow.

本说明介绍当前使用的模型训练流程。`旧的代码/` 中的脚本仅作历史参考；当前流程请运行 `训练主程序/` 中的脚本。

PyTorch is not used by these workflows. The classifiers are LightGBM, XGBoost, CatBoost and Random Forest.

本流程不使用 PyTorch；分类器为 LightGBM、XGBoost、CatBoost 和随机森林。

## Workflow | 流程

1. **Prepare the inputs.** Keep all seven sample-polygon sets in their year-pair folders. Add the optical-image pairs, DEM and daily GPM rasters to the example folders shown below; these folders contain instructions only, not source data. | **准备输入数据。** 7 期样本面按年份对放在对应文件夹中；光学影像、DEM 和逐日 GPM 栅格需放入下方列出的示例目录。示例目录目前只有说明文件，不包含原始数据。
2. **Train the classifiers.** `训练主程序.py` (main training script) builds annual features, trains LightGBM, XGBoost, CatBoost and Random Forest for each image pair, and saves model and validation outputs. | **训练分类器。** `训练主程序.py` 构建各年度特征，并对每个年份对分别训练 LightGBM、XGBoost、CatBoost 和随机森林，保存模型及验证结果。
3. **Create the annual landslide maps.** After training, `加权预测结果.py` (weighted prediction and ensemble script) combines the four model predictions with equal weights and writes one probability raster for each year pair. | **生成年度滑坡图。** 训练完成后运行 `加权预测结果.py`，对 4 个分类器的预测结果进行等权集成，为每个年份对输出一幅概率栅格。
4. **Run optional diagnostics.** `阈值敏感性分析.py` (threshold-sensitivity analysis script) evaluates probability thresholds, while `SHAP_interp.py` produces model-interpretation outputs. | **运行可选诊断。** `阈值敏感性分析.py` 评估概率阈值；`SHAP_interp.py` 生成模型解释结果。

## Input layout | 输入目录

```text
Input/Training/
├── SamplePolygons/
│   ├── 18-19/landslide_samples_18-19.shp (+ companion files)
│   ├── 19-20/landslide_samples_19-20.shp (+ companion files)
│   ├── 20-21/landslide_samples_20-21.shp (+ companion files)
│   ├── 21-22/landslide_samples_21-22.shp (+ companion files)
│   ├── 22-23/landslide_samples_22-23.shp (+ companion files)
│   ├── 23-24/landslide_samples_23-24.shp (+ companion files)
│   └── 24-25/landslide_samples_24-25.shp (+ companion files)
└── ExampleData/
    ├── OpticalImagery/SR2_<year>*/<image>.tif
    ├── Terrain/dem.tif
    └── Precipitation/GPM/<year>/*.tif
```

The optical-image directories are selected by the `SR2_<year>*` pattern. The GPM files are read from directories named for their calendar year. Keep the sample shapefile sidecars together; do not upload only the `.shp` component.

光学影像目录按 `SR2_<年份>*` 模式读取；GPM 栅格按日历年份放入对应文件夹。Shapefile 的配套文件需完整保留，不能只复制 `.shp`。

## Run | 运行

From the repository root in PowerShell:

在仓库根目录打开 PowerShell，运行：

```powershell
cd "代码\机器学习训练\训练主程序"
python "训练主程序.py"
python "加权预测结果.py"
```

Optional analyses can be run after the training outputs are available:

训练结果生成后，可按需运行：

```powershell
python "阈值敏感性分析.py"
python "SHAP_interp.py"
```

## Code map | 代码结构

```text
训练主程序/
├── 训练主程序.py                 # Train and evaluate the four classifiers
├── 加权预测结果.py               # Equal-weight ensemble and annual probability maps
├── 阈值敏感性分析.py             # Probability-threshold sensitivity analysis
├── SHAP_interp.py                # SHAP interpretation
└── utils/
    ├── base_utils.py             # Input discovery, alignment and label rasterization
    ├── label_process.py          # Training-table construction and spatial folds
    ├── Optical_feature.py        # Optical indices and change features
    ├── Terrain_feature.py        # Terrain predictors
    ├── Precipitation_feature.py # GPM-derived predictors
    ├── Plotting.py               # Training and evaluation plots
    └── head.py / __init__.py     # Shared imports and project configuration
```

## Outputs | 输出

- Model files, shared training tables and validation outputs are saved to `Output/TrainingResults/Models/`. | 模型文件、共享训练表和验证结果写入 `Output/TrainingResults/Models/`。
- Ensemble probability rasters (one per year pair) are saved to `Output/LandslideProducts/Data/annual_probability_maps/`. | 每个年份对的等权集成概率图写入 `Output/LandslideProducts/Data/annual_probability_maps/`。
- Threshold-sensitivity and SHAP results are saved to `Output/TrainingResults/`. | 阈值敏感性分析和 SHAP 结果写入 `Output/TrainingResults/`。
