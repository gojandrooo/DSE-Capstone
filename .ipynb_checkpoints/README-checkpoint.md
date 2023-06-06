# DSE-Capstone

to run this project:
- open terminal window and clone the repository (git clone ...url)
- open a terminal window in the repository folder
- in terminal, run the following command:
```python

> conda create -n venv_debris_flow python=3.10.10
> conda activate venv_debris_flow
> conda -y install jupyter jupyterlab ipykernel
> python -m ipykernel install --user --name=venv_debris_flow
> # you may need to restart the environment to use it in jupyter lab
> conda deactivate
> conda activate venv_debris_flow
> pip install -r requirements.txt
```
    - it will install the required packages with necessary dependencies found in requirements.txt

now when you launch a notebook, select the `venv_debris_flow` kernel


Decription of data files:

- [file_name.ext]
    - [file description]

- The Staley et al. (2016) model report and data:
 
    https://pubs.er.usgs.gov/publication/ofr20161106
 
- LF20_F40_220.csv
    The LANDFIRE Scott and Burgan fire behavior model (used for deriving vegetation and fuel-related features):
 
    https://landfire.gov/fbfm40.php
    csv direct download: https://landfire.gov/CSV/LF2020/LF20_F40_220.csv
 
- Geological maps of US states:
 
    https://mrdata.usgs.gov/geology/state/
 
- USGS 3D elevation program:
 
    https://www.usgs.gov/3d-elevation-program
 
  Dataset with 1/3 arc-seconds (~10 meters) resolution,  e.g. individual tile located at 35°N / 119°W):
  https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n35w119/USGS_13_n35w119.tif