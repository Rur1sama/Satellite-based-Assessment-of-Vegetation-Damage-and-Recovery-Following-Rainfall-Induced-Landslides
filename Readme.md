# Satellite-based Assessment of Vegetation Damage and Recovery Following Rainfall-Induced Landslides

[English](Readme.md) | [简体中文](Readme_zh.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23447197.svg)](https://doi.org/10.5281/zenodo.23447197)

This repository contains the code and selected project products for mapping rainfall-induced landslides and assessing vegetation damage and recovery in Jiangwan Town, southern China. The workflow has two main parts: landslide model training and ecological-impact analysis. The project covers seven annual image pairs (2018–2019 to 2024–2025).

## 1. Configure the environment

The project was developed with Python 3.9. Create an environment and install the dependencies:

```bash
conda create -n landslide-recovery python=3.9
conda activate landslide-recovery
python -m pip install -r requirements.txt
```

PyTorch is not used or required. The mapping models are LightGBM, XGBoost, CatBoost and Random Forest.

## 2. Place the data in the corresponding folders

The seven manually interpreted sample-polygon sets are included. Other source inputs are not bundled; the English-named folders below show where to place them. Each sample shapefile is a set of companion files (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg` and `.qmd`), not a standalone `.shp` file.

```text
.
├── Input/
│   ├── Training/
│   │   ├── SamplePolygons/
│   │   │   ├── 18-19/landslide_samples_18-19.shp (+ companion files)
│   │   │   ├── 19-20/landslide_samples_19-20.shp (+ companion files)
│   │   │   ├── 20-21/landslide_samples_20-21.shp (+ companion files)
│   │   │   ├── 21-22/landslide_samples_21-22.shp (+ companion files)
│   │   │   ├── 22-23/landslide_samples_22-23.shp (+ companion files)
│   │   │   ├── 23-24/landslide_samples_23-24.shp (+ companion files)
│   │   │   └── 24-25/landslide_samples_24-25.shp (+ companion files)
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
├── code/ # Workflow source code
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

The ecological-analysis configuration reads FVC, LAI and GPP from `Output/EcologicalProducts/`; the study-area boundary and slope raster are read from `Input/EcologicalAnalysis/ExampleData/`. The SNAP graph and project configuration are kept with the image-processing code under `code/Sentinel-2 Image Processing/SNAP_base/`; Sentinel-2 source scenes remain under `Input/EcologicalAnalysis/ExampleData/Sentinel2L2A/`. FAPAR is retained as a product but is disabled as an analysis outcome in the example configuration. VOD is also disabled by default. See the guides for [model training](<code/Model training and landslide mapping/Readme.md>), [ecological analysis](<code/Ecological-impact and recovery analysis/Readme.md>) and [Sentinel-2 processing](<code/Sentinel-2 Image Processing/README.md>). Chinese guides are available as [model training](<code/Model training and landslide mapping/Readme_zh.md>) and [ecological analysis](<code/Ecological-impact and recovery analysis/Readme_zh.md>).

## 3. Run the workflows

Run the two parts separately from their respective code directories.

**Model training and landslide mapping**

```powershell
cd "code\Model training and landslide mapping\Training workflow"
python "Main.py"
python "weighted_ensemble.py"
```

The first script trains the four classifiers across the seven image pairs. The second combines their predictions and writes the annual probability maps. See the [model-training guide](<code/Model training and landslide mapping/Readme.md>) for optional threshold-sensitivity and SHAP commands.

**Ecological-impact and recovery analysis**

```powershell
cd "code\Ecological-impact and recovery analysis"
python run_pipeline.py --help
python run_pipeline.py --config config_example.json --steps all
```

The help command lists the available stages and options. The [ecological-analysis guide](<code/Ecological-impact and recovery analysis/Readme.md>) reproduces those options and explains each stage.

## 4. Find the outputs

- The landslide inventory and seven annual probability maps are in `Output/LandslideProducts/Data/`.
- Model fits, validation plots, threshold analyses and related intermediate results are written under `Output/TrainingResults/`.
- Ecological-analysis tables and figures are written under `Output/EcologicalAnalysis/`.
- Eight-date FVC, LAI, FAPAR and GPP rasters are retained locally under `Output/EcologicalProducts/` and excluded from Git. The associated code-and-data release is archived on Zenodo ([DOI: 10.5281/zenodo.23447197](https://doi.org/10.5281/zenodo.23447197)). See the Zenodo record's file list for the exact products included in the archived release.

For future release updates, confirm redistribution rights and metadata for all products and third-party inputs. See [LICENSE](LICENSE) for code licensing information.
