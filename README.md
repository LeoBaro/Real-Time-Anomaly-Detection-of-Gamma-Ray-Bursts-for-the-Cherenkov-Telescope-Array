# phd

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