#!/bin/bash
export PYTHONPATH=/data01/homes/baroncelli/phd/cta-sag-sci:/data01/homes/baroncelli/phd/sag-sci
export CTOOLS=/data01/homes/baroncelli/.conda/envs/bphd
nohup jupyter notebook --ip='*' --port=8890 --no-browser --allow-root &
