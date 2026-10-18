# 基于卫星的降雨诱发滑坡植被损伤与恢复评估

[English](Readme.md) | [简体中文](Readme_zh.md)

本项目提供中国南方江湾镇的降雨诱发滑坡制图，以及滑坡后植被损伤与恢复评估代码。工作流程分为两部分：滑坡模型训练和生态影响分析；研究覆盖 2018–2019 至 2024–2025 的 7 个年度影像对。

## 1. 配置环境

项目原有说明记录的 Python 版本为 3.9。创建环境并安装依赖：

```bash
conda create -n landslide-recovery python=3.9
conda activate landslide-recovery
python -m pip install -r requirements.txt
```

本项目不使用或依赖 PyTorch。滑坡制图使用 LightGBM、XGBoost、CatBoost 和随机森林模型。

## 2. 将数据放入对应位置

仓库已包含 7 期人工判读样本面；其他原始输入数据不随代码仓库提供。以下英文目录只展示数据应放置的位置。每个 Shapefile 都由多个配套文件组成（`.shp`、`.shx`、`.dbf`、`.prj`、`.cpg` 和 `.qmd`），不能只放 `.shp` 文件。

```text
.
├── Input/
│   ├── Training/
│   │   ├── SamplePolygons/
│   │   │   ├── 18-19/landslide_samples_18-19.shp（及配套文件）
│   │   │   ├── 19-20/landslide_samples_19-20.shp（及配套文件）
│   │   │   ├── 20-21/landslide_samples_20-21.shp（及配套文件）
│   │   │   ├── 21-22/landslide_samples_21-22.shp（及配套文件）
│   │   │   ├── 22-23/landslide_samples_22-23.shp（及配套文件）
│   │   │   ├── 23-24/landslide_samples_23-24.shp（及配套文件）
│   │   │   └── 24-25/landslide_samples_24-25.shp（及配套文件）
│   │   └── ExampleData/
│   │       ├── OpticalImagery/SR2_<year>*/
│   │       ├── Terrain/dem.tif
│   │       └── Precipitation/GPM/<year>/
│   └── EcologicalAnalysis/ExampleData/
│       ├── Sentinel2L2A/
│       ├── Climate_ERA5Land/
│       ├── Climate_GLASS_PAR/
│       ├── StudyArea/study_area_boundary.shp
│       ├── Terrain/slope.tif
│       └── VOD/
├── Output/
│   ├── LandslideProducts/Data/
│   │   ├── annual_probability_maps/landslide_probability_<pair>.tif
│   │   └── landslide_event_inventory.gpkg
│   ├── EcologicalProducts/
│   │   ├── FVC/*.tif
│   │   ├── LAI/*.tif
│   │   ├── FAPAR/*.tif
│   │   └── GPP/*.tif
│   ├── TrainingResults/
│   └── EcologicalAnalysis/
├── code/
│   ├── Model training and landslide mapping/
│   │   ├── Training workflow/
│   │   └── Legacy scripts/
│   ├── Ecological-impact and recovery analysis/
│   └── Sentinel-2 Image Processing/
│       ├── SNAP_base/
│       │   ├── Graphs/myBiophysical10mOp.xml
│       │   ├── SR2_LAI_cal.xml
│       │   └── SR2_LAI_cal.session.snap
│       └── run_lai_batch.py
├── package.json
├── requirements.txt
└── LICENSE
```

生态分析配置会从 `Output/EcologicalProducts/` 读取 FVC、LAI 和 GPP；从 `Input/EcologicalAnalysis/ExampleData/` 读取研究区边界和坡度栅格。SNAP 图和项目配置统一保存在影像处理代码目录 `code/Sentinel-2 Image Processing/SNAP_base/`；Sentinel-2 原始影像仍放在 `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/`。FAPAR 虽已保存在产品目录中，但示例配置默认不将其作为分析指标；VOD 也默认关闭。运行前请阅读[模型训练说明](<code/Model training and landslide mapping/Readme_zh.md>)、[生态分析说明](<code/Ecological-impact and recovery analysis/Readme_zh.md>)和[Sentinel-2 影像处理说明](<code/Sentinel-2 Image Processing/README.md>)。

## 3. 运行流程

两部分需要分别在各自的代码目录中运行。

**模型训练与滑坡制图**

```powershell
cd "code\Model training and landslide mapping\Training workflow"
python "Main.py"
python "weighted_ensemble.py"
```

第一个脚本对 7 个年度影像对训练 4 种分类器；第二个脚本对分类结果进行等权集成，并生成年度概率图。阈值敏感性分析和 SHAP 分析等可选脚本见[模型训练说明](<code/Model training and landslide mapping/Readme_zh.md>)。

**生态影响与恢复分析**

```powershell
cd "code\Ecological-impact and recovery analysis"
python run_pipeline.py --help
python run_pipeline.py --config config_example.json --steps all
```

`--help` 会显示可用的分析阶段和参数；完整参数解释和各阶段说明见[生态分析说明](<code/Ecological-impact and recovery analysis/Readme_zh.md>)。

## 4. 查看输出结果

- 滑坡矢量清单和 7 期年度概率图位于 `Output/LandslideProducts/Data/`。
- 模型、验证图、阈值分析和中间结果写入 `Output/TrainingResults/`。
- 生态分析表格和图件写入 `Output/EcologicalAnalysis/`。
- FVC、LAI、FAPAR、GPP 各 8 期栅格保存在本地 `Output/EcologicalProducts/`。这些大体量栅格已从 Git 中排除，应通过配套公共数据集发布。本次 README 更新没有将数据上传到公共仓库。

公开发布前，请确认产品和第三方输入数据的再分发许可及元数据。代码许可见 [LICENSE](LICENSE)。
