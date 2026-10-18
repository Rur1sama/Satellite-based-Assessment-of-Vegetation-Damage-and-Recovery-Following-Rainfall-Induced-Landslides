# Ecological-impact and recovery analysis

[English](Readme.md) | [简体中文](Readme_zh.md)

This module estimates landslide-associated changes in FVC, LAI and GPP and summarizes post-landslide recovery. Its configuration and outputs are separate from the model-training workflow.

## Workflow

1. **Prepare inputs.** Place the required FVC, LAI and GPP rasters in `Output/EcologicalProducts/`.
2. **Map landslide units (`events`).** Convert annual probability maps into landslide patches using the configured threshold.
3. **Define zones (`zones`).** Build landslide cores, nearby buffers and reference areas.
4. **Extract indicators (`panel`).** Summarize FVC, LAI and GPP for each analysis unit and year.
5. **Estimate changes and make figures (`did`, `plots`).** Compare affected areas with reference trends and produce configured size and recovery summaries.

## Data locations

```text
Input/EcologicalAnalysis/ExampleData/
├── StudyArea/study_area_boundary.shp (+ companion files)
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

Place the study-area boundary and slope raster at the example paths. The seven annual probability maps are included in the project; ecological indicator rasters must be supplied separately and are not included in the code repository. The example configuration uses FVC, LAI and GPP; FAPAR and VOD are disabled by default. Edit `config_example.json` to change paths or enable another indicator.

## Sentinel-2 products

This pipeline reads precomputed ecological rasters; it does not run SNAP itself. To regenerate Sentinel-2-derived LAI, FVC or FAPAR, place source scenes in `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/` and run from the repository root:

```powershell
python "code\Sentinel-2 Image Processing\run_lai_batch.py"
```

The SNAP graph and project settings are in `code/Sentinel-2 Image Processing/SNAP_base/`. The batch script uses `SNAP_base/Graphs/myBiophysical10mOp.xml` by default. See the [Sentinel-2 processing guide](<../Sentinel-2 Image Processing/README.md>).

## Command-line help

Run from this directory:

```powershell
python run_pipeline.py --help
```

The command-line interface is:

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

`--config` defaults to the `config_example.json` beside the script; `--steps` defaults to `all`. Provide one or more stage names after `--steps`. Use `--force` to recompute cached intermediate results.

## Run

```powershell
python run_pipeline.py --config config_example.json --steps all
```

Selected stages:

```powershell
python run_pipeline.py --config config_example.json --steps events zones panel did plots
```

Force recomputation:

```powershell
python run_pipeline.py --config config_example.json --steps panel did plots --force
```

## Code map

```text
Ecological-impact and recovery analysis/
├── run_pipeline.py                 # Workflow entry point and CLI
├── config_example.json             # Input, output and analysis settings
├── calculate_daily_gpp_vpm.py      # Daily GPP estimation utility
└── ecoservice_analysis/
    ├── events.py                   # Probability rasters to mapped patches
    ├── zones.py                    # Impact and reference zones
    ├── panel.py                    # Annual indicator extraction
    ├── did.py                      # Change estimation
    ├── core_size.py                # Landslide-size analysis
    ├── recovery_figures.py         # Recovery summaries and figures
    ├── area_response.py            # Area-level response summaries
    ├── summaries.py                # Summary tables
    ├── plotting.py                 # General plots
    ├── individual_diagnostics.py   # Unit-level diagnostics
    ├── binning.py                  # Grouping utilities
    ├── io_utils.py                 # Path and file utilities
    └── config.py                   # Configuration loading
```

## Outputs

Results are written under `Output/EcologicalAnalysis/`: `01_landslide_events.gpkg` (mapped patches), `02_landslide_zones.gpkg` (impact and reference areas), `03_zone_outcome_panel.csv` (annual indicator values), `04_did_results/` (estimates and tables), and `05_figures/` (figures).
