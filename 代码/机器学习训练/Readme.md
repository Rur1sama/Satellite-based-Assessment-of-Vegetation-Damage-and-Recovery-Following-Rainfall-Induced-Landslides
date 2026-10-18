# Model training and landslide mapping

[English](Readme.md) | [简体中文](Readme_zh.md)

This guide describes the current workflow for training landslide classifiers and producing annual probability maps. Scripts in `旧的代码/` are retained for reference; use `训练主程序/` for the current workflow. PyTorch is not used or required. The classifiers are LightGBM, XGBoost, CatBoost and Random Forest.

## Workflow

1. **Prepare inputs.** Keep all seven sample-polygon sets in their year-pair folders. Place the optical-image pairs, DEM and daily GPM rasters in the example folders below; these placeholder folders contain instructions only.
2. **Train classifiers.** Run `训练主程序.py` to build annual features, train the four classifiers for each image pair, and save model and validation outputs.
3. **Create annual landslide maps.** Run `加权预测结果.py` after training to combine the four model predictions with equal weights and write one probability raster per year pair.
4. **Run optional diagnostics.** `阈值敏感性分析.py` evaluates probability thresholds; `SHAP_interp.py` produces model-interpretation outputs.

## Input layout

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

Optical imagery is selected using the `SR2_<year>*` pattern; GPM rasters are read from folders named by calendar year. Keep all shapefile companion files together.

## Run

From the repository root in PowerShell:

```powershell
cd "代码\\机器学习训练\\训练主程序"
python "训练主程序.py"
python "加权预测结果.py"
```

Optional analyses, run from the same directory after training:

```powershell
python "阈值敏感性分析.py"
python "SHAP_interp.py"
```

## Code map

```text
训练主程序/
├── 训练主程序.py                 # Train and evaluate the classifiers
├── 加权预测结果.py               # Equal-weight ensemble and annual maps
├── 阈值敏感性分析.py             # Threshold sensitivity
├── SHAP_interp.py                # SHAP interpretation
└── utils/
    ├── base_utils.py             # Input discovery, alignment and labels
    ├── label_process.py          # Training table and spatial folds
    ├── Optical_feature.py        # Optical indices and change features
    ├── Terrain_feature.py        # Terrain predictors
    ├── Precipitation_feature.py  # GPM predictors
    └── Plotting.py               # Training and evaluation plots
```

## Outputs

- Models, shared training tables and validation outputs: `Output/TrainingResults/Models/`.
- Annual probability rasters: `Output/LandslideProducts/Data/annual_probability_maps/`.
- Threshold-sensitivity and SHAP results: `Output/TrainingResults/`.
