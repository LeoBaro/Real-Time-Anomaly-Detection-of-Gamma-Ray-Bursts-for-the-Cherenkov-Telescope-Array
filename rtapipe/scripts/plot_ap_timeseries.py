import argparse
import multiprocessing
from pathlib import Path
from functools import partial

from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

def parse_params(filename):
    return {
        "runid":   filename.split("runid_")[1].split("_trial_")[0],
        "trial":   filename.split("trial_")[1].split("_simtype_")[0],
        "simtype": filename.split("simtype_")[1].split("_onset_")[0],
        "onset":   float(filename.split("onset_")[1].split("_delay_")[0]),
        "delay":   float(filename.split("delay_")[1].split("_offset_")[0]),
        "offset":  float(filename.split("offset_")[1].split("_itype_")[0]),
        "itype":   filename.split("itype_")[1].split("_itime_")[0],
        "itime":   filename.split("itime_")[1].split("_normalized_")[0],
        "normalized": filename.split("normalized_")[1].split(".csv")[0]
    }
    

# python plot_ap_timeseries.py -f /data01/homes/baroncelli/AP_DATA_10000/ap_data_bkg_T_10_TSL_10/dataset_trials_10000_type_bkg_tobs_100/integration_te_integration_time_10_region_radius_0.2_timeseries_lenght_10/
def make_plot(file_path, outputdir):
    filename = file_path.name   
    print(f"Processing: {filename}")
    try:
        params = parse_params(filename)
    except:
        params = {
            "onset": None,
            "simtype": "bkg",
            "itype": "te",
            "normalized": "True"
        }
        print(f"Error parsing filename: {filename}")
    # extract the information from the file name: runid_run0406_ID000126_trial_00000002_simtype_grb_onset_25_delay_0_offset_0.5_itype_t_itime_1_normalized_False.csv
    plot = PhotometrySinglePlot(params)
    _ = plot.addData(file_path)        
    _ = plot.plotScatterSingleAxes()
    plot.save(outputdir, f"{filename}")
    plot.destroy()

        


if __name__=='__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("--dir", "-d", type=str, required=True)
        parser.add_argument("--output-dir", "-o", type=str, required=True)
        parser.add_argument("--limit", "-l", type=int, required=False, default=-1)
        args = parser.parse_args()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        files = list(Path(args.dir).glob('*.csv'))[:args.limit]

        print(f"Found {len(list(files))} files")

        func = partial(make_plot, outputdir=args.output_dir)

        with multiprocessing.Pool(20) as p:
            p.map(func, files)


