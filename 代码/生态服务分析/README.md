# Ecological-impact and recovery analysis | 生态影响与恢复分析

This module estimates landslide-associated changes in FVC, LAI and GPP and summarizes post-landslide recovery. Its configuration and outputs are kept separate from the model-training workflow.

本模块用于估计滑坡相关的 FVC、LAI 和 GPP 变化，并汇总滑坡后的恢复情况。配置文件和分析结果与模型训练部分分开放置。

## Workflow | 分析流程

1. **Prepare the ecological inputs.** Ensure that the required FVC, LAI and GPP rasters are available in `Output/EcologicalProducts/`. If the Sentinel-2-derived LAI or FVC products need to be regenerated, run the SNAP workflow described below before starting the analysis. | **准备生态分析输入。** 确认 `Output/EcologicalProducts/` 中已有分析所需的 FVC、LAI 和 GPP 栅格。若需重新生成基于 Sentinel-2 的 LAI 或 FVC 产品，请先按下文说明运行 SNAP 流程。
2. **Map analysis units (`events`).** Convert the annual landslide probability maps into mapped landslide patches using the configured threshold. | **生成滑坡分析单元（`events`）。** 按配置的概率阈值，将年度滑坡概率图转换为滑坡斑块。
3. **Define impact and reference areas (`zones`).** Build the landslide cores, nearby buffer zones and reference areas used in the analysis. | **划定影响区和参考区（`zones`）。** 生成滑坡核心区、邻近缓冲区和分析所需的参考区域。
4. **Extract ecological measurements (`panel`).** Summarize FVC, LAI and GPP for each analysis unit and year. | **提取生态指标（`panel`）。** 汇总各分析单元逐年的 FVC、LAI 和 GPP。
5. **Estimate changes and make figures (`did`, `plots`).** Compare changes in landslide-affected areas with reference trends, then produce summary tables and figures, including the size and recovery analyses configured for this project. | **估计变化并生成图件（`did`、`plots`）。** 将滑坡影响区的变化与参考趋势比较，再输出汇总表和图件，包括配置的规模差异及恢复分析。

## Data locations | 数据位置

```text
Input/EcologicalAnalysis/ExampleData/
├── StudyArea/study_area_boundary.shp (+ companion files)
└── Terrain/slope.tif

Output/
├── LandslideProducts/Data/
│   └── annual_probability_maps/landslide_probability_<pair>.tif
└── EcologicalProducts/
    ├── FVC/FVC_<date>_*.tif
    ├── LAI/LAI_<date>_*.tif
    ├── FAPAR/FAPAR_<date>_*.tif
    └── GPP/GPP_<date>_existing_epsilon_kgC_m2_day.tif
```

Place the study-area boundary and slope raster at the example paths. The ecological rasters and landslide probability maps are already staged in the project. The example configuration uses FVC, LAI and GPP; FAPAR and VOD are disabled by default. Edit `config_example.json` if you change file locations or enable another indicator.

研究区边界和坡度栅格需放在上述路径。生态指标栅格与滑坡概率图已暂存于项目中。示例配置启用 FVC、LAI 和 GPP；FAPAR 与 VOD 默认关闭。如需更改文件位置或启用其他指标，请编辑 `config_example.json`。

## Upstream Sentinel-2 products | Sentinel-2 上游产品

This analysis pipeline reads precomputed FVC, LAI and GPP rasters; it does not run SNAP itself. To regenerate the Sentinel-2-derived LAI, FVC or FAPAR products, place the source scenes in `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/` and run `python "代码\哨兵2号影像处理\run_lai_batch.py"` from the repository root. The SNAP graph and project settings now reside with the image-processing code at `代码/哨兵2号影像处理/SNAP_base/`; the batch script uses `SNAP_base/Graphs/myBiophysical10mOp.xml` by default. See the [Sentinel-2 processing guide](../哨兵2号影像处理/README.md) for setup and outputs.

本分析流程直接读取预先生成的 FVC、LAI 和 GPP 栅格，不会自行调用 SNAP。若需重新反演 Sentinel-2 的 LAI、FVC 或 FAPAR，请将原始影像放入 `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/`，并在仓库根目录运行 `python "代码\哨兵2号影像处理\run_lai_batch.py"`。SNAP 图和项目参数现统一保存在影像处理代码目录 `代码/哨兵2号影像处理/SNAP_base/` 中；批处理脚本默认调用 `SNAP_base/Graphs/myBiophysical10mOp.xml`。环境配置和输出说明见 [Sentinel-2 影像处理指南](../哨兵2号影像处理/README.md)。

## Command-line help | 命令行帮助

Run commands from this directory after installing the project dependencies:

安装依赖后，在当前目录运行：

```powershell
python run_pipeline.py --help
```

The command-line interface defined by `run_pipeline.py` is:

`run_pipeline.py` 中定义的命令行选项如下：

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

`--config` defaults to the `config_example.json` beside the script, and `--steps` defaults to `all`. Use one or more stage names after `--steps`; `--force` recomputes intermediate results instead of reusing cached files.

`--config` 默认指向脚本同目录下的 `config_example.json`；`--steps` 默认是 `all`。`--steps` 后可填写一个或多个阶段名称；添加 `--force` 可强制重新计算中间结果，而不复用缓存。

## Run | 运行

Run the complete workflow:

运行完整流程：

```powershell
python run_pipeline.py --config config_example.json --steps all
```

Run selected stages (later stages may require outputs from earlier stages):

仅运行指定阶段（后续阶段可能依赖前序阶段的输出）：

```powershell
python run_pipeline.py --config config_example.json --steps events zones panel did plots
```

Force recomputation:

强制重新计算：

```powershell
python run_pipeline.py --config config_example.json --steps panel did plots --force
```

## Code map | 代码结构

```text
生态服务分析/
├── run_pipeline.py                 # Workflow entry point and command-line options
├── config_example.json              # Input, output and analysis settings
├── calculate_daily_gpp_vpm.py      # Daily GPP estimation utility
└── ecoservice_analysis/
    ├── events.py                    # Annual probability rasters to mapped patches
    ├── zones.py                     # Impact and reference zones
    ├── panel.py                     # Annual ecological indicator extraction
    ├── did.py                       # Change estimation
    ├── core_size.py                 # Small- and large-landslide analysis
    ├── recovery_figures.py          # Recovery summaries and figures
    ├── area_response.py             # Area-level response summaries
    ├── summaries.py                 # Summary tables
    ├── plotting.py                  # General analysis plots
    ├── individual_diagnostics.py    # Diagnostics for individual units
    ├── binning.py                   # Grouping utilities
    ├── io_utils.py                  # Path and file utilities
    ├── config.py                    # Configuration loading
    └── head.py / __init__.py        # Shared imports and module initialization
```

## Outputs | 输出

Results are written under `Output/EcologicalAnalysis/`:

分析结果写入 `Output/EcologicalAnalysis/`：

- `01_landslide_events.gpkg` — mapped landslide patches.
- `02_landslide_zones.gpkg` — impact and reference areas.
- `03_zone_outcome_panel.csv` — extracted annual indicator values.
- `04_did_results/` — change estimates and summary tables.
- `05_figures/` — figures and visual summaries.

- `01_landslide_events.gpkg`：滑坡斑块矢量数据。
- `02_landslide_zones.gpkg`：影响区和参考区。
- `03_zone_outcome_panel.csv`：提取的年度指标值。
- `04_did_results/`：变化估计和汇总表。
- `05_figures/`：图件和可视化汇总。
