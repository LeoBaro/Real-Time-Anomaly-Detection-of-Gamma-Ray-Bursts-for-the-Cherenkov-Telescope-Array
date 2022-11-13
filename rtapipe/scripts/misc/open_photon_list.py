import argparse

from astropy.io import fits

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, help="")
    args = parser.parse_args()

    with fits.open(args.file) as hdul:
        header = hdul[1].header
        data = hdul[1].data
        columns = hdul[1].columns
        pointing = header["RA_PNT"], header["DEC_PNT"]
        print(pointing)
        print("Min RA: ", data["RA"].min())
        print("Min DEC: ", data["DEC"].min())
        print("Max RA: ", data["RA"].max())
        print("Max DEC: ", data["DEC"].max())

        print(columns)
        print(data)

