import os

from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from lib.plots import showSkymap


if "DATA" not in os.environ:

    print("Please, set $DATA.")

else:

    for root, subdirs, files in os.walk(os.path.join(os.path.expandvars(os.environ["DATA"]), "obs")):

        skymaps_files = [f for f in files if ".skymap.fits" in f]

        for filename in skymaps_files: 

            fits_file = os.path.join(root, filename)

            png_output = fits_file.replace(".fits",".png") 
        
            print(f"Producing.. {png_output}")
    
            #if not os.path.isfile(png_output):
                
            showSkymap(fits_file, show=False, tex=False)

            #else:
            #    print(f"File already exists!")

            