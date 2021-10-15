import argparse

from astropy.io import fits

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, help="")
    args = parser.parse_args()

    with fits.open(args.file) as hdul:
        header = hdul[1].header
        print(header)
        data = hdul[1].data
        columns = hdul[1].columns
        print(columns)
        print(data)

