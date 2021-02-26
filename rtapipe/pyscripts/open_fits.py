import argparse
from astropy.io import fits

# python RTAscience/phlistInfo.py --phtlist ~/phd/DATA/obs/2021-01-31_17-07-58/run0406_ID000126/ebl000001.fits

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phtlist', type=str, required=True)
args = parser.parse_args()

with fits.open(args.phtlist) as hdul:

    # print(hdul.info())

    """
    for i,hdu in enumerate(hdul):
        print(f"\nHDU {i}")
        hdr = hdu.header
        print(repr(hdr))
    """

    hdr = hdul["EVENTS"].header
    # print(repr(hdr))

    data = hdul["EVENTS"].data
    print(data[0:3])
    print(f"Number of events: {len(data)}")
    print(f"Energy mean: {data.field('ENERGY').mean()}")
    print(f"RA mean: {data.field('RA').mean()}")
    print(f"DEC mean: {data.field('DEC').mean()}")    