# Sentinel-2 image processing | 哨兵2号影像处理

This module contains the SNAP graph and project settings, together with the scripts for deriving LAI, FVC and FAPAR from Sentinel-2 L2A scenes. Raw scenes are not included; place them in `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/`. The graph and project configuration are kept beside the processing code in `SNAP_base/`.

本模块集中存放 SNAP 图、项目参数以及 Sentinel-2 L2A 影像的 LAI、FVC 和 FAPAR 反演脚本。原始影像不随仓库提供，请放入 `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/`；图文件和项目配置统一保存在本目录的 `SNAP_base/` 中。

## Files | 文件说明

- `run_lai_batch.py`: runs the SNAP graph in batch, then exports the LAI, FVC and FAPAR bands. | 批量运行 SNAP 图，并拆分导出 LAI、FVC 和 FAPAR 波段。
- `SNAP_base/Graphs/myBiophysical10mOp.xml`: SNAP processing graph used by the batch script. | 批处理脚本调用的 SNAP 处理图。
- `SNAP_base/SR2_LAI_cal.xml` and `SR2_LAI_cal.session.snap`: SNAP project/session configuration; product paths point to the repository's input and output layout. | SNAP 项目及会话配置；产品路径对应本仓库的输入、输出目录结构。

## Run | 运行

Install SNAP with the Sentinel-2 Biophysical processor and make the Graph Processing Tool (`gpt`) available on `PATH`. From the repository root, run:

安装 SNAP 及 Sentinel-2 生物物理处理器，并确保 Graph Processing Tool（`gpt`）可通过 `PATH` 调用。在仓库根目录运行：

```powershell
python "代码\哨兵2号影像处理\run_lai_batch.py"
```

If `gpt` or the graph file is in a non-default location, set `SNAP_GPT_PATH` or `SNAP_GRAPH_XML`. Multiband intermediates are written to `Output/EcologicalProducts/SNAP_Multiband/`; single-band products are written beneath `Output/EcologicalProducts/`.

如果 `gpt` 或 SNAP 图文件不在默认位置，可分别设置 `SNAP_GPT_PATH` 或 `SNAP_GRAPH_XML`。多波段中间结果写入 `Output/EcologicalProducts/SNAP_Multiband/`；单波段产品写入 `Output/EcologicalProducts/` 下对应的指标目录。
