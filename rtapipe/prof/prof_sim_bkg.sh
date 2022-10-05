#!/bin/bash

python -m cProfile -o sim_bkg.prof ../../cta-sag-sci/RTAscience/simBkg.py -f ./config.yml -out ./sim_bkg_output