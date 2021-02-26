import re
from math import sqrt
from numpy import square
import os.path, argparse
import pandas as pd
import matplotlib.pyplot as plt

from astro.lib.photometry import Photometrics
from RTAscience.lib.RTAUtils import get_pointing
from RTAscience.cfg.Config import Config

from rtapipe.pyscripts.data_utils import DataUtils

plt.rcParams.update({'font.size': 22})

class Photometry:

    def __init__(self):
        pass

    def main(self, args):

        cfg = Config(os.path.join(args.obs_dir, "config.yaml"))
        datapath = cfg.get('data')
        runid = cfg.get('runid')
        simtype = cfg.get('simtype')  # 'grb' -> src+bkg; 'bkg' -> empty fields
        time_windows = []

        # trials = cfg.get('trials')  # trials

        if args.window_size == 0:
            raise ValueError("window-size must be greater than zero.")

        if args.mode == "cumulative":
            t_start = 0
            t_cumul = 0
            t_stop = cfg.get('tobs')
            while t_start + t_cumul < t_stop:
                t_cumul += args.window_size
                time_windows.append((t_start, t_start + t_cumul))

        elif args.mode == "windowed":
            t_start = 0
            t_stop = cfg.get('tobs')
            w_start = 0
            while w_start + args.window_size < t_stop:
                time_windows.append((w_start, w_start + args.window_size))
                w_start += args.window_step

        input_file = None
        if simtype == 'grb':
            input_file = os.path.join(args.obs_dir, runid, "ebl000001.fits")
            output_dir = os.path.join(args.obs_dir, runid, "ap")
            output_file = os.path.join(output_dir, f"ebl000001_ap_mode")   
        elif simtype == 'bkg': # bkg
            input_file = os.path.join(args.obs_dir, "backgrounds", "bkg000001.fits")
            output_dir = os.path.join(args.obs_dir, "backgrounds", "ap")
            output_file = os.path.join(output_dir, f"bkg000001_ap_mode")

        try: 
            os.mkdir(output_dir) 
        except OSError as error: 
            pass 


        if args.mode == "cumulative":
            output_file += f"_{args.mode}_wsize_{args.window_size}_rad_{args.region_radius}.csv"
        elif args.mode == "windowed":
            output_file += f"_{args.mode}_wsize_{args.window_size}_wstep_{args.window_step}_rad_{args.region_radius}.csv"




        template =  os.path.join(datapath, f'templates/{runid}.fits')
        pointing = get_pointing(template)

        phm = Photometrics({ 'events_filename': input_file })
        region = {
            'ra': pointing[0],
            'dec': pointing[1],
        }

        # print('File: ', input_file)
        print('\nRegion center ', region, 'with radius', args.region_radius, 'deg')
        
        total = 0
        if os.path.isfile(output_file) and args.override == 0:
            print(f"File {output_file} already exists!")
        else:
            with open(output_file, "w") as of:
                of.write("TMIN,TMAX,WINDOW_CENTER,COUNTS,ERROR\n")    
                for time_window in time_windows:
                    region_count = phm.region_counter(region, args.region_radius, tmin=time_window[0], tmax=time_window[1])
                    total += region_count
                    # print(f'tmin {time_window[0]} tmax {time_window[1]} -> counts: {region_count}')
                    of.write(f"{time_window[0]},{time_window[1]},{(time_window[1]+time_window[0])/2},{region_count},{sqrt(region_count)}\n")
            print("Produced: ",output_file)
        
        if args.countcheck == 1:
            allcounts = phm.region_counter(region, args.region_radius, tmin=0, tmax=cfg.get("tobs"))
            print(f"Count check ! allcounts: {allcounts}, cumulative windows count: {total}")
            assert total == allcounts

        if args.plot == 1:

            png_output_file = output_file.replace('.csv', '.png') 

            if os.path.isfile(png_output_file) and args.override == 0:
                print(f"File {png_output_file} already exists!")
            else:

                df = pd.read_csv(output_file)

                fig = plt.figure(figsize=(30,15))

                plt.title(f'Aperture Photometry Counts (rad: {args.region_radius}° (ra:{region["ra"]}, dec:{region["dec"]}))')

                if args.mode == "cumulative":
                    x = df['TMAX']
                    plt.ylabel('Cumulative Counts')
                    plt.xlabel('TMAX (tmin=0)')
                    plt.suptitle(f'Cumulative Mode. window size: {args.window_size}')

                elif args.mode == "windowed":
                    x = df['WINDOW_CENTER']
                    plt.ylabel('Windowed Counts')
                    plt.xlabel(f'Window center')
                    plt.suptitle(f'Windowed Mode [0-{cfg.get("tobs")}]. window size: {args.window_size}, window step: {args.window_step}')

                y = df['COUNTS']

                yerr = df['ERROR']


                # read from filename
                onsetVal = int(re.search('os_(.+?)_emin', input_file).group(1))
                plt.axvline(x=onsetVal, color="red", linestyle="--")


                plt.scatter(x,y)
                plt.errorbar(x, y, yerr=yerr, fmt="o") 

                # add emission
                # emissionData = getTheoreticalEmissionModel("run0406_ID000126")

                fig.savefig(png_output_file)

                if args.showplot == 1:
                    plt.show()

                plt.close()
                print("Produced: ", png_output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="produces aperture photometry plots")
    parser.add_argument('-obs', '--obs-dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('-rad', '--region-radius', help='the region radius (default: 0.2°)', default=0.2, type=float)
    parser.add_argument('-md', '--mode', choices=["cumulative", "windowed"], help='The ', type=str)
    parser.add_argument('-wsize', '--window-size', help='The window size (seconds)', type=int, required=True)
    parser.add_argument('-wstep', '--window-step', help='The window step (seconds). Requires if --mode = windowed', type=int, required=False, default=1)
    parser.add_argument('-pl', '--plot', help='Produce plot', type=int, default=0, required=False)
    parser.add_argument('-ov', '--override', help='Ovverride data and plots', type=int, default=0, required=False)
    parser.add_argument('-cc', '--countcheck', help='Check the number of the counts', type=int, default=0, required=False)
    parser.add_argument('-sp', '--showplot', help='', type=int, default=0, required=False)
    args = parser.parse_args()

    p = Photometry()
    p.main(args)