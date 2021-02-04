"""
Usage:

  # src only
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-42_st_grb_tr_1_os_0 -md windowed -wsize 50 -wstep 25 -rad 1 -pl 1
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-42_st_grb_tr_1_os_0 -md cumulative -wsize 50 -rad 1 -pl 1

  # bkg+src
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-58-36_st_grb_tr_1_os_600/ -md windowed -wsize 50 -wstep 25 -rad 1 -pl 1
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-58-36_st_grb_tr_1_os_600/ -md cumulative -wsize 50 -rad 1 -pl 1

  # bkg only
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-38_st_bkg_tr_1_os_0/ -md windowed -wsize 50 -wstep 25 -rad 1 -pl 1
  python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/2021-02-04_10-41-38_st_bkg_tr_1_os_0/ -md cumulative -wsize 50 -rad 1 -pl 1
 

"""
import os.path, argparse
from astro.lib.photometry import Photometrics
from RTAscience.lib.RTAUtils import get_pointing
from RTAscience.cfg.Config import Config
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def main(args):

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


    if simtype == 'grb':
        input_file = os.path.join(args.obs_dir, runid, "ebl000001.fits")
        output_file = os.path.join(args.obs_dir, runid, f"ebl000001_ap_mode")   
    elif simtype == 'bkg': # bkg
        input_file = os.path.join(args.obs_dir, "backgrounds", "bkg000001.fits")
        output_file = os.path.join(args.obs_dir, "backgrounds", f"bkg000001_ap_mode")

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

    print('File: ', input_file)
    print('Region center ', region, 'with radius', args.region_radius, 'deg')
    
    with open(output_file, "w") as of:
        of.write("TMIN,TMAX,COUNTS\n")    
        for time_window in time_windows:
            region_count = phm.region_counter(region, args.region_radius, tmin=time_window[0], tmax=time_window[1])
            print(f'tmin {time_window[0]} tmax {time_window[1]} -> counts: {region_count}')
            of.write(f"{time_window[0]},{time_window[1]},{region_count}\n")

    print("Produced: ",output_file)

    if args.plot == 1:
        df = pd.read_csv(output_file)

        fig = plt.figure(figsize=(15,15))

        plt.title(f'Aperture Photometry Counts (rad: {args.region_radius}° (ra:{region["ra"]}, dec:{region["dec"]}))')

        if args.mode == "cumulative":
            x = df['TMAX']
            plt.ylabel('Cumulative Counts')
            plt.xlabel('TMAX (tmin=0)')
            plt.suptitle(f'Cumulative Mode. window size: {args.window_size}')

        elif args.mode == "windowed":
            x = df['TMIN']
            plt.ylabel('Windowed Counts')
            plt.xlabel(f'TMIN (- TMIN + {args.window_size})')
            plt.suptitle(f'Windowed Mode. window size: {args.window_size}, window step: {args.window_step}')

        y = df['COUNTS']


        plt.plot(x,y)
        output_file = output_file.replace('.csv', '.png') 
        fig.savefig(output_file)
        plt.close()
        print("Produced: ",output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="produces aperture photometry plots")
    parser.add_argument('-obs', '--obs-dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('-rad', '--region-radius', help='the region radius (default: 0.2°)', default=0.2, type=float)
    parser.add_argument('-md', '--mode', choices=["cumulative", "windowed"], help='The ', type=str)
    parser.add_argument('-wsize', '--window-size', help='The window size (seconds)', type=int, required=True)
    parser.add_argument('-wstep', '--window-step', help='The window step (seconds). Requires if --mode = windowed', type=int, required=False, default=1)
    parser.add_argument('-pl', '--plot', help='Produce plot', type=int, default=0, required=False)

    args = parser.parse_args()
    main(args)