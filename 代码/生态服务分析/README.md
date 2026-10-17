# 滑坡生态服务 DID 分析代码

这套代码用于把年度滑坡掩膜和 LAI/GPP/VOD 栅格组织成面板数据，并估计滑坡对植被的直接破坏、周边缓冲区破坏辐射范围、事件研究恢复轨迹和滑坡前异常信号。

## 核心思路

代码没有使用传统双向固定效应 DID，而是用更适合交错事件的组-时间 ATT：

`ATT(g, t) = [滑坡年份为 g 的区域在 t 与 g-1 的变化] - [同期尚未滑坡或稳定控制区域的变化]`

这样做的作用是扣除同一年气候、区域背景变化等共同扰动，估计“如果这里没有发生滑坡，本来应该怎样”的反事实。

## 文件结构

- `config_example.json`：路径、年份、缓冲区、DID 参数配置。
- `run_pipeline.py`：一键运行入口。
- `ecoservice_analysis/events.py`：把 7 个年度滑坡 mask tif 转成滑坡事件面。
- `ecoservice_analysis/zones.py`：生成滑坡体内部、外部缓冲带和远距离稳定控制带。
- `ecoservice_analysis/panel.py`：提取每个空间单元每年的 LAI/GPP/VOD 均值。
- `ecoservice_analysis/did.py`：组-时间 DID 和事件研究估计。
- `ecoservice_analysis/summaries.py`：破坏辐射范围、可选恢复阈值和前兆信号汇总。
- `ecoservice_analysis/plotting.py`：输出事件研究、辐射范围、生态损失和 cohort 追踪图。

## 使用步骤

1. 复制或直接修改 `config_example.json`。
2. 把 `landslide.masks` 中 7 个滑坡掩膜路径改成你的真实 tif。
3. 把 `outcomes` 中 LAI、GPP、VOD 的 `path_template` 改成真实年度栅格路径。
4. 在当前目录运行：

```powershell
python run_pipeline.py --config config_example.json --steps all
```

也可以分步运行：

```powershell
python run_pipeline.py --config config_example.json --steps events zones panel did plots
```

## 输出结果

默认输出到 `E:/python_base/Slide_dect/Output/EcoService_DID`：

- `01_landslide_events.gpkg`：滑坡事件矢量面，每个连通斑块一个事件。
- `02_landslide_zones.gpkg`：滑坡体、0-20m、20-50m 等缓冲区和稳定控制带。
- `03_zone_outcome_panel.csv`：面板数据，后续所有 DID 都基于它。
- `04_did_results/group_time_att.csv`：每个滑坡年份组、每个相对年份的 ATT。
- `04_did_results/event_study_att.csv`：按相对年份汇总后的事件研究结果。
- `04_did_results/radiation_range_summary.csv`：每个结果变量在灾后年份的最大显著破坏距离。
- `05_figures`：图件。

## 关键参数建议

- `spillover_rings_m`：10m Sentinel-2 LAI 建议从 `[0,20]`、`[20,50]`、`[50,100]` 开始；MODIS GPP/VOD 分辨率较粗，缓冲区需要更大才稳定。
- `control_group`：默认 `not_yet_plus_never`，同时使用“尚未滑坡区域”和远距离稳定控制带；如果想更贴近采矿论文的 not-yet-treated 设计，改成 `not_yet`。
- `reference_relative_year`：默认 `-1`，即滑坡发生前一年作为基准。
- `event_year`：18-19 掩膜默认记为 2019 年发生，19-20 记为 2020 年，以此类推。

## 结果怎么解释

- `att < 0`：滑坡造成了额外 LAI/GPP/VOD 损失。
- 某个缓冲区在灾后 `rel_year = 0,1,2` 显著为负：说明破坏辐射到该距离。
- event-study ATT 在 `rel_year = 0` 后逐渐回到 0：表示 DID 校正后的生态功能恢复过程。
- 灾前 `rel_year < 0` 已显著为负：可能是失稳前植被胁迫或水分异常信号，也可能表示平行趋势不充分，需要谨慎解释。
