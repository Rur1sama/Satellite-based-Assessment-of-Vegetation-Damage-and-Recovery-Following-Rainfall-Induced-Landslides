# Training sample polygons

This directory contains the manually interpreted sample polygons used for model development. Each subdirectory corresponds to one consecutive Sentinel-2 image pair:

| Folder | Image pair |
|---|---|
| `18-19` | 2018–2019 |
| `19-20` | 2019–2020 |
| `20-21` | 2020–2021 |
| `21-22` | 2021–2022 |
| `22-23` | 2022–2023 |
| `23-24` | 2023–2024 |
| `24-25` | 2024–2025 |

Each folder contains a Shapefile dataset (`.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`; the available QGIS metadata sidecar is also retained). The training code expects one `.shp` file in the corresponding year-pair folder. Attribute names and label values are preserved from the interpreted source data.
