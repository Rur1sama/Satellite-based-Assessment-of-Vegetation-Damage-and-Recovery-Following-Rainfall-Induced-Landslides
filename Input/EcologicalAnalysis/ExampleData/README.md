# Ecological-analysis input examples

The ecological workflow reads the derived indicator rasters from `Output/EcologicalProducts/`. Source and ancillary data are not included; use the directories below as local input locations and retain their English names so the relative code paths remain valid.

- `Sentinel2L2A/`: source Sentinel-2 L2A scenes used by SNAP.
- SNAP graph and project configuration are maintained with the image-processing code under `code/Sentinel-2 Image Processing/SNAP_base/`; `run_lai_batch.py` uses that graph by default. Set `SNAP_GRAPH_XML` only when using a different graph.
- `Climate_ERA5Land/`: ERA5-Land temperature and vapour-pressure-deficit inputs used for GPP estimation.
- `Climate_GLASS_PAR/`: GLASS PAR files, arranged by year with `hdf/` and `xml/` subfolders.
- `StudyArea/`: the boundary file `study_area_boundary.shp`.
- `Terrain/`: the slope raster expected by the recovery configuration as `slope.tif`.
- `VOD/`: optional VOD rasters; this outcome is disabled in the supplied example configuration.
