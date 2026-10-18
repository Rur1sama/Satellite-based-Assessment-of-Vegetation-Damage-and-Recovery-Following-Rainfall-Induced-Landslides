# 生态影响与恢复分析

[English](Readme.md) | [简体中文](Readme_zh.md)

本模块用于估计滑坡相关的 FVC、LAI 和 GPP 变化，并汇总滑坡后的恢复情况。其配置文件和分析结果与模型训练流程分开放置。

## 工作流程

1. **准备输入。** 将分析所需的 FVC、LAI 和 GPP 栅格放入 `Output/EcologicalProducts/`。
2. **生成滑坡分析单元（`events`）。** 按配置的概率阈值，将年度概率图转换为滑坡斑块。
3. **划定分区（`zones`）。** 构建滑坡核心区、邻近缓冲区和参考区域。
4. **提取生态指标（`panel`）。** 汇总各分析单元逐年的 FVC、LAI 和 GPP。
5. **估计变化并生成图件（`did`、`plots`）。** 将滑坡影响区与参考趋势比较，并生成配置的规模差异和恢复分析结果。

## 数据位置

```text
Input/EcologicalAnalysis/ExampleData/
├── StudyArea/study_area_boundary.shp（及配套文件）
└── Terrain/slope.tif

Output/
├── LandslideProducts/Data/
│   ├── annual_probability_maps/landslide_probability_<pair>.tif
│   └── landslide_event_inventory.gpkg
└── EcologicalProducts/
    ├── FVC/FVC_<date>_*.tif
    ├── LAI/LAI_<date>_*.tif
    ├── FAPAR/FAPAR_<date>_*.tif
    └── GPP/GPP_<date>_existing_epsilon_kgC_m2_day.tif
```

研究区边界和坡度栅格需放在上述示例路径中。项目已包含 7 期年度概率图；生态指标栅格不包含在代码仓库中，需要另行提供。示例配置启用 FVC、LAI 和 GPP；FAPAR 与 VOD 默认关闭。如需更改文件路径或启用其他指标，请编辑 `config_example.json`。

## Sentinel-2 产品

本流程直接读取预先生成的生态栅格，不会自行运行 SNAP。若需重新反演 Sentinel-2 的 LAI、FVC 或 FAPAR，请将原始影像放入 `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/`，并在仓库根目录运行：

```powershell
python "code\Sentinel-2 Image Processing\run_lai_batch.py"
```

SNAP 图和项目参数位于 `code/Sentinel-2 Image Processing/SNAP_base/`；批处理脚本默认调用 `SNAP_base/Graphs/myBiophysical10mOp.xml`。详见 [Sentinel-2 影像处理说明](<../Sentinel-2 Image Processing/README.md>)。

## 命令行帮助

在当前目录运行：

```powershell
python run_pipeline.py --help
```

命令行选项如下：

```text
usage: run_pipeline.py [-h] [--config CONFIG] [--steps STEPS [STEPS ...]] [--force]

Run landslide ecosystem-service DID analysis.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to JSON config.
  --steps STEPS [STEPS ...]
                        Steps to run: all, events, zones, panel, did, plots.
  --force               Force recomputation instead of using cached intermediate files.
```

`--config` 默认指向脚本同目录下的 `config_example.json`；`--steps` 默认是 `all`。可在 `--steps` 后填写一个或多个阶段名称；添加 `--force` 可强制重新计算缓存的中间结果。

## 运行

```powershell
python run_pipeline.py --config config_example.json --steps all
```

仅运行指定阶段：

```powershell
python run_pipeline.py --config config_example.json --steps events zones panel did plots
```

强制重新计算：

```powershell
python run_pipeline.py --config config_example.json --steps panel did plots --force
```

## 代码结构

```text
Ecological-impact and recovery analysis/
├── run_pipeline.py                 # 工作流程入口和命令行参数
├── config_example.json             # 输入、输出和分析设置
├── calculate_daily_gpp_vpm.py      # 日尺度 GPP 估算工具
└── ecoservice_analysis/
    ├── events.py                   # 概率图转为滑坡斑块
    ├── zones.py                    # 影响区和参考区
    ├── panel.py                    # 年度生态指标提取
    ├── did.py                      # 变化估计
    ├── core_size.py                # 滑坡规模分析
    ├── recovery_figures.py         # 恢复汇总和图件
    ├── area_response.py            # 区域尺度响应汇总
    ├── summaries.py                # 汇总表
    ├── plotting.py                 # 通用图件
    ├── individual_diagnostics.py   # 单元诊断
    ├── binning.py                  # 分组工具
    ├── io_utils.py                 # 路径和文件工具
    └── config.py                   # 配置读取
```

## 输出

结果写入 `Output/EcologicalAnalysis/`：`01_landslide_events.gpkg`（滑坡斑块）、`02_landslide_zones.gpkg`（影响区和参考区）、`03_zone_outcome_panel.csv`（年度指标值）、`04_did_results/`（估计结果和表格）及 `05_figures/`（图件）。
