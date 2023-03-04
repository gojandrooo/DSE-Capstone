# Data preparation notebooks

This directory contains the following notebooks for data preparation.

1. `add_site_ids.ipynb`: Retrieves Staley data and adds site IDs.  The result is stored in the Parquet file `staley16_debrisflow.parquet`.
2. `extract_contributing_region.ipynb`: Computes catchment area and fuel related features for each site.  Reads `staley16_debrisflow.parquet` and writes `staley16_observations_catchment_fuelpars_v3.parquet`.

