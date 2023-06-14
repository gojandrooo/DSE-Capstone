# UCSD Jacobs School of Engineering 
## Masters in Data Science and Engineering
### Capstone Project: Post-Fire Debris Flow Likelihood Prediction
- Contributors:
    - Alejandro Hohmann
    - Bhanu Muvva
    - Chunxia Tong
- Digital Object Identifier: https://doi.org/10.6075/J0PG1RZH

## Project Abstract
*Debris Flows are a distinct type of landslide that suddenly occur without warning. They are fast-moving channels of water and soil that carry large natural objects like boulders and trees, or human-made objects including cars. In the American West, Debris Flows have directly caused death and property damage. Debris Flows often occur after rain events and the burn scars left behind by wildfires increase their likelihood. Given the increasing frequency of extreme weather events, it is critical to predict Debris Flows and take precautionary action before they occur. This project builds upon prior research of predicting Debris Flows using additional geological features and more advanced machine learning techniques. The project also includes an intuitive interface for decision makers to access these probability estimates.*


## Using this Repository
This repository is organized so that everything in the `notebooks` folder can be run sequentially. However, the data preparation steps are already latent in this repository and do not need to be executed to use. 
- To skip straight to the final output, use the `app` folder. This folder was copied to EC2 instance for end-user access. It can also be run locally by simply executing the `app.ipynb` or `app.py` files.
- To experiment with NN architecture, use the `notebooks/03_ML_models` folder.

```
├── app
    ├── data
    ├── model
    ├── app.ipynb
    └── app.py
├── data
├── images
├── Landfire
├── notebooks
    ├── 00_archive
    ├── 01_staley
    ├── 02_data_prep
    ├── 03_ML_models
    └── 04_data_viz
├── ABC Debris - Final Presentation.pdf
├── ABC Debris - Final Report.pdf
├── LICENSE
├── README.md
├── requirements.txt
└── venv.py
```  


#### Dependencies managed with virtual environment and `requirements.txt` file:
- open terminal window and clone the repository (git clone ...url)
- open a terminal window in the repository folder
- to create a virtual environment and install the required packages with necessary dependencies, run the following commands in terminal:
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

*when you launch a notebook, make sure to select the `venv_debris_flow` kernel*



## Prior Work Referenced

- The Staley et al. (2016) model report and data:
    - https://pubs.er.usgs.gov/publication/ofr20161106