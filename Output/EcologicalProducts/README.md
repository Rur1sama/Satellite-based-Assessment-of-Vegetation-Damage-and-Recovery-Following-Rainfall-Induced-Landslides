# Ecological indicator products

This local data-staging directory contains the selected analysis rasters:

- `FVC/`, `LAI/` and `FAPAR/`: eight single-band Sentinel-2/SNAP products for the observation dates 2018-10-07, 2019-12-06, 2020-11-10, 2021-12-10, 2022-12-25, 2023-11-20, 2024-11-29 and 2025-11-21.
- `GPP/`: the eight matching daily GPP rasters named with the `existing_epsilon` variant used by the ecological-analysis configuration.

These derived rasters are large and are ignored by Git. The associated code-and-data release is archived on Zenodo ([DOI: 10.5281/zenodo.23447197](https://doi.org/10.5281/zenodo.23447197)); consult its file list to confirm which rasters are included. Do not add these large files to the code repository. The SNAP multiband intermediate files and SNAP software/cache files are intentionally not included.
