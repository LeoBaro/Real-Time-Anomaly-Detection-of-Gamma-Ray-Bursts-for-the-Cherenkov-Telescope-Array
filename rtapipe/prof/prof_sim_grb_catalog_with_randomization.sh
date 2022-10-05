#!/bin/bash

#python -m cProfile -o sim_bkg.prof ../../cta-sag-sci/RTAscience/simBkg.py -f ./config_trials_1-10.yaml -out ./output

python3 -m cProfile -o sim_grb_catalog_with_randomization.prof ../../cta-sag-sci/RTAscience/simGRBcatalogWithRandomization.py -f config.yml --output-dir ./sim_grb_catalog_with_randomization_output --print no --mp-threads 30
