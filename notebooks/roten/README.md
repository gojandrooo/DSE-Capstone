# PostFireDebrisFlow

This repository contains the notebooks for post-fire debris flow models published in the proceedings of the 2022 IEEE Big Data Conference:

"Machine Learning for Improved Post-fire Debris Flow Likelihoods Prediction"

by Daniel Roten, Jessica Block, Daniel Crawl, Jenny Lee, and Ilkay Altintas

## Prerequisites

- Pandas, including Geopandas
- PostgreSQL with PostGIS module
- xarray with rioxarray extension
- Dask
- Scikit-learn
- Distributed computing environment. The Nautilus Kubernetes cluster was used in the examples.

## Notebooks

- The data preparation notebook [add_site_ids.ipynb](data_preparation/add_site_ids.ipynb) creates an unique Site ID and adds it as additonal column.
- The Dask notebook [


## Data sources:

- The Staley et al. (2016) model report and data:
 
	https://pubs.er.usgs.gov/publication/ofr20161106
 
- The LANDFIRE Scott and Burgan fire behavior model (used for deriving vegetation and fuel-related features):
 
	https://landfire.gov/fbfm40.php
 
- Geological maps of US states:
 
	https://mrdata.usgs.gov/geology/state/
 
- USGS 3D elevation program:
 
	https://www.usgs.gov/3d-elevation-program
 
  Dataset with 1/3 arc-seconds (~10 meters) resolution,  e.g. individual tile located at 35°N / 119°W):
  https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n35w119/USGS_13_n35w119.tif
