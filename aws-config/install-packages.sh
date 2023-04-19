#!/bin/bash
# This script installs a multiple pip packages on a SageMaker Studio Kernel Application
# packages copied from requirements.txt file
# would be nice to have a this alternative but AWS doesn't seem to work this way
'''
pip install -r requirements.txt
'''

set -eux

pip install pandas==1.4.2
pip install sklearn==1.0.2
pip install torch==1.13.1
pip install tabula==2.6.0
pip install s3fs==2023.1.0
pip install shapely==2.0.1
pip install geopandas==0.12.2
pip install pyarrow==11.0.0
pip install pysheds==11.0.0
pip install fastfuels==1.0.4
pip install rioxarray==0.13.4