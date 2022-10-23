import os
import argparse
from RTAscience.lib.RTACtoolsAnalysis import RTACtoolsAnalysis
from RTAscience.lib.RTAVisualise import  plotSkymap

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--subtraction', type=str, choices=["NONE","IRF"], required=True)
    args = parser.parse_args()

    rta = RTACtoolsAnalysis()
    rta.usepnt = True
    rta.sky_subtraction = args.subtraction

    for root, subdirs, files in os.walk(args.data_dir):

        fits_files = [f for f in files if ".fits" in f and ".skymap" not in f]

        for filename in fits_files: 

            rta.input = os.path.join(root, filename)
            rta.output = rta.input.replace(".fits",f"_sub_{rta.sky_subtraction}.skymap.fits")

            print(f"Producing.. {rta.output}")

            if not os.path.isfile(rta.output):
                rta.run_skymap(wbin=0.02)
            else:
                print(f"File {rta.output} already exists!")
                
            plotSkymap(rta.output, show=False, usetex=False, png=root)
        

if __name__=="__main__":
    main()
