import argparse
import multiprocessing
from pathlib import Path
from functools import partial

from rtapipe.lib.plotting.APPlot import APPlot
from rtapipe.lib.rtapipeutils.FileSystemUtils import parse_params


# python plot_ap_timeseries.py -f /data01/homes/baroncelli/AP_DATA_10000/ap_data_bkg_T_10_TSL_10/dataset_trials_10000_type_bkg_tobs_100/integration_te_integration_time_10_region_radius_0.2_timeseries_lenght_10/
def make_plot(file_path, start, points, outputdir, maxflux=None):
    filename = file_path.name   
    print(f"Processing: {filename}")
    try:
        params = parse_params(filename)
    except:
        params = {
            "onset": None,
            "simtype": "bkg",
            "itype": "te",
            "normalized": "True",
            "offset" : 0
        }
        print(f"Error parsing filename: {filename}")
    params["maxflux"] = maxflux
    # extract the information from the file name: runid_run0406_ID000126_trial_00000002_simtype_grb_onset_25_delay_0_offset_0.5_itype_t_itime_1_normalized_False.csv
    applot = APPlot()
    applot.plot(file_path, params, start=start, lenght=points)
    applot.save(outputdir, f"{filename}")
    print(f"Saved: {outputdir}/{filename}")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True)
    parser.add_argument("-s", "--start", type=int, required=False, default=0)
    parser.add_argument("-l", "--length", type=int, required=False, default=None)
    parser.add_argument("-mf", "--max-flux", type=int, required=False, default=None)
    parser.add_argument("-o", "--output-dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if Path(args.dir).is_dir():
        print("Processing directory")
        # search all .csv inside the args.dir and its subdirectories
        files = list(Path(args.dir).rglob("*.csv"))
        print(f"Found {len(list(files))} files in {Path(args.dir)}")

        func = partial(make_plot, start=args.start, points=args.length, outputdir=args.output_dir, maxflux=args.max_flux)

        with multiprocessing.Pool(20) as p:
            p.map(func, files)

    if Path(args.dir).is_file():
        print("Processing single file")
        make_plot(Path(args.dir), args.length, args.output_dir, args.max_flux)

if __name__=='__main__':
    main()

