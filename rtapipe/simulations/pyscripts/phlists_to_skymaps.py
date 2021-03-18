import os, argparse

from astropy.io import fits

from RTAscience.lib.RTACtoolsAnalysis import RTACtoolsAnalysis
from RTAscience.cfg.Config import Config

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--subtraction', type=str, choices=["NONE","IRF"], required=True)
    parser.add_argument('--override', type=int, choices=[1,0], required=False, default=0)
    args = parser.parse_args()

    if "DATA" not in os.environ:

        print("Please, set $DATA.")

    else:

        rta = RTACtoolsAnalysis()
        rta.usepnt = True
        rta.sky_subtraction = args.subtraction

        for root, subdirs, files in os.walk(os.path.join(os.path.expandvars(os.environ["DATA"]), "obs", args.datadir)):

            fits_files = [f for f in files if ".fits" in f and ".skymap" not in f]

            for filename in fits_files: 

                rta.input = os.path.join(root, filename)
                rta.output = rta.input.replace(".fits",f"_sub_{rta.sky_subtraction}.skymap.fits")
            
                configPath = os.path.join(root, "config.yaml")
                # print("Found config: ",os.path.join(root, "config.yaml"))

                cfg = Config(configPath)
                rta.configure(cfg)

                print(f"Producing.. {rta.output}")

                if not os.path.isfile(rta.output) or args.override == 1:
                    rta.run_skymap(wbin=0.02)
                else:
                    print(f"File {rta.output} already exists!")
                    