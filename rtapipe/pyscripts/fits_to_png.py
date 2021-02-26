import os, argparse
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from lib.plots import showSkymap

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--type', type=str, choices=["sm","lc"], required=True)
    parser.add_argument('--override', type=int, choices=[1,0], required=False, default=0)
    args = parser.parse_args()

    if "DATA" not in os.environ:

        print("Please, set $DATA.")

    else:

        for root, subdirs, files in os.walk(os.path.join(os.path.expandvars(os.environ["DATA"]), "obs")):

            if args.type == "sm":
                fits_files = [f for f in files if ".skymap.fits" in f]
            else:
                fits_files = [f for f in files if ".lightcurve.fits" in f]

            for filename in fits_files: 

                fits_file = os.path.join(root, filename)

                png_output = fits_file.replace(".fits",".png") 
            
                print(f"Producing.. {png_output}")
        
                if not os.path.isfile(png_output) or args.override == 1:
                    
                    if args.type == "sm":
                        showSkymap(fits_file, show=False, tex=False)
                    else:
                        showLightCurve(fits_file, show=False, tex=False)

                else:
                    print(f"File already exists!")

                