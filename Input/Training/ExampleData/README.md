# Training input examples

Raw predictor datasets are not included. Place local copies under the matching English-named directories below; paths are resolved from the repository root.

- `OpticalImagery/`: annual Sentinel-2 surface-reflectance GeoTIFFs in date-named folders such as `SR2_20181007/`.
- `Terrain/`: the DEM expected by the training scripts as `dem.tif`.
- `Precipitation/GPM/`: GPM daily precipitation rasters arranged in year subfolders (`2018/` … `2025/`).
- `StudyArea/`: optional study-area boundary, named `study_area_boundary.shp` with its Shapefile sidecars.

Do not commit the actual source rasters here. Add data-source, date, resolution, CRS and licence details to the associated data record.
