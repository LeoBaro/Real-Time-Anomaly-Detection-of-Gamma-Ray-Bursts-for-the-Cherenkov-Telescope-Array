import argparse
from sagsci.tools.fits import Fits
from sagsci.tools.plotting import SkyImage
from RTAscience.lib.RTAUtils import get_pointing


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-ff", "--fits-file", type=str, required=True)
    parser.add_argument("-t", "--template", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()

    plot = SkyImage()
    pointing = get_pointing(args.template)
    plot.set_pointing(pointing[0], pointing[1])
    plot.counts_map(file=args.fits_file, name=args.output)

