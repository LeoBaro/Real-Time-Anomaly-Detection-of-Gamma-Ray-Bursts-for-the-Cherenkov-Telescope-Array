#!/bin/bash

python -m cProfile -o prof_out.prof ../../cta-sag-sci/RTAscience/simBkg.py -f ./config_trials_1-10.yaml -out ./output