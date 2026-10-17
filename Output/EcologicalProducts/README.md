# Ecological indicator products

This local data-staging directory contains the selected analysis rasters:

- `FVC/`, `LAI/` and `FAPAR/`: eight single-band Sentinel-2/SNAP products for the observation dates 2018-10-07, 2019-12-06, 2020-11-10, 2021-12-10, 2022-12-25, 2023-11-20, 2024-11-29 and 2025-11-21.
- `GPP/`: the eight matching daily GPP rasters named with the `existing_epsilon` variant used by the ecological-analysis configuration.

These derived rasters are large and are ignored by Git. They are retained locally for deposit in the separate public data record; do not add them to the code repository. The SNAP multiband intermediate files and SNAP software/cache files are intentionally not included.
