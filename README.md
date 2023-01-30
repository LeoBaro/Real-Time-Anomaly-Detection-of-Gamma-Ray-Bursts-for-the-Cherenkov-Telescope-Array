# Real-Time Anomaly Detection of Gamma-Ray Bursts for the Cherenkov Telescope Array using Deep Learning

## Installation

### Environment for deep learning
Install `conda-lock` then,
```bash
git clone --recurse-submodule git@github.com:LeoBaro/phd.git
conda-lock install --name YOURENV conda-lock.yml
```

### Environment for data generation
```bash
conda env create --name <envname> python=3.8
conda config --env --add channels cta-observatory
conda install ctools=1.7.4 pyaml pandas tqdm matplotlib scikit-learn
```

### Start Juyter server 
Install the anaconda environment within the ipykernel (do it once):
```bash
python -m ipykernel install --user --name phd-tf
```
Deactivate and reactivate the environment, then start the server with:
```bash
jupyter notebook --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```