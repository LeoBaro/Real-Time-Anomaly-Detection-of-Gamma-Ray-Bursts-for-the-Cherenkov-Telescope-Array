#!/bin/bash

python -m cProfile -o sim_bkg.prof ../../cta-sag-sci/RTAscience/simBkg.py -f ./config_trials_1-10.yaml -out ./output

python -m cProfile -o sim_grb_catalog_with_randomization.prof python ../../cta-sag-sci/RTAscience/simGRBcatalogWithRandomization.py -f config.yml --output-dir ./output --print no --mp-threads 30