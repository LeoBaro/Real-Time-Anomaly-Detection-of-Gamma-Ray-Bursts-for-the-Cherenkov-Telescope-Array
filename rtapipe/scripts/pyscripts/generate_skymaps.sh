#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# python $DIR/pyscripts/phlists_to_skymaps.py --override 1 --subtraction NONE --datadir autoencoder
python $DIR/pyscripts/phlists_to_skymaps.py --override 1 --subtraction IRF --datadir autoencoder
python $DIR/pyscripts/fits_to_png.py --type sm --datadir autoencoder