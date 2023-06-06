conda create -n venv_debris_flow python=3.10.10
conda activate venv_debris_flow
conda -y install jupyter jupyterlab ipykernel
python -m ipykernel install --user --name=venv_debris_flow
# you may need to restart the environment to use it in jupyter lab
conda deactivate
conda activate venv_debris_flow
pip install -r requirements.txt