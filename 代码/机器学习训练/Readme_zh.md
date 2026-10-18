# 模型训练与滑坡制图

[English](Readme.md) | [简体中文](Readme_zh.md)

本说明介绍当前使用的滑坡分类器训练和年度概率图生成流程。`旧的代码/` 中的脚本仅作参考；当前流程请使用 `训练主程序/`。本流程不使用或依赖 PyTorch，分类器为 LightGBM、XGBoost、CatBoost 和随机森林。

## 工作流程

1. **准备输入数据。** 将 7 期样本面放在对应年份对文件夹中，并将光学影像、DEM 和逐日 GPM 栅格放入下方示例目录；这些占位目录仅含说明，不含原始数据。
2. **训练分类器。** 运行 `训练主程序.py` 构建年度特征，并针对每个影像年份对训练 4 种分类器，保存模型及验证结果。
3. **生成年度滑坡图。** 训练完成后运行 `加权预测结果.py`，对 4 个分类器的预测结果进行等权集成，为每个年份对输出一幅概率栅格。
4. **运行可选诊断。** `阈值敏感性分析.py` 评估概率阈值；`SHAP_interp.py` 生成模型解释结果。

## 输入目录

```text
Input/Training/
├── SamplePolygons/
│   ├── 18-19/landslide_samples_18-19.shp（及配套文件）
│   ├── 19-20/landslide_samples_19-20.shp（及配套文件）
│   ├── 20-21/landslide_samples_20-21.shp（及配套文件）
│   ├── 21-22/landslide_samples_21-22.shp（及配套文件）
│   ├── 22-23/landslide_samples_22-23.shp（及配套文件）
│   ├── 23-24/landslide_samples_23-24.shp（及配套文件）
│   └── 24-25/landslide_samples_24-25.shp（及配套文件）
└── ExampleData/
    ├── OpticalImagery/SR2_<year>*/<image>.tif
    ├── Terrain/dem.tif
    └── Precipitation/GPM/<year>/*.tif
```

光学影像目录按 `SR2_<年份>*` 模式读取；GPM 栅格按日历年份放入对应文件夹。Shapefile 的配套文件需完整保留。

## 运行

在仓库根目录打开 PowerShell：

```powershell
cd "代码\\机器学习训练\\训练主程序"
python "训练主程序.py"
python "加权预测结果.py"
```

训练完成后，可在同一目录运行可选分析：

```powershell
python "阈值敏感性分析.py"
python "SHAP_interp.py"
```

## 代码结构

```text
训练主程序/
├── 训练主程序.py                 # 训练并评估分类器
├── 加权预测结果.py               # 等权集成并生成年度概率图
├── 阈值敏感性分析.py             # 阈值敏感性分析
├── SHAP_interp.py                # SHAP 模型解释
└── utils/
    ├── base_utils.py             # 输入发现、对齐和标签处理
    ├── label_process.py          # 训练表构建和空间交叉验证分块
    ├── Optical_feature.py        # 光学指数和变化特征
    ├── Terrain_feature.py        # 地形预测因子
    ├── Precipitation_feature.py  # GPM 降水预测因子
    └── Plotting.py               # 训练和评估图件
```

## 输出

- 模型、共享训练表和验证结果：`Output/TrainingResults/Models/`。
- 年度概率栅格：`Output/LandslideProducts/Data/annual_probability_maps/`。
- 阈值敏感性分析和 SHAP 结果：`Output/TrainingResults/`。
