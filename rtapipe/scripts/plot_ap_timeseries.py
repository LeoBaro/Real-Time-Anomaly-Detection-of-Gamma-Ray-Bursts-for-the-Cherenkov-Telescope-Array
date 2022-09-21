import argparse
from pathlib import Path

from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

# python plot_ap_timeseries.py -f /data01/homes/baroncelli/AP_DATA_10000/ap_data_bkg_T_10_TSL_10/dataset_trials_10000_type_bkg_tobs_100/integration_te_integration_time_10_region_radius_0.2_timeseries_lenght_10/

if __name__=='__main__':

        parser = argparse.ArgumentParser()
        parser.add_argument("--dir", "-d", type=str, required=True)
        parser.add_argument("--output-dir", "-od", type=str, required=True)
        args = parser.parse_args()

        for file in Path(args.dir).glob('*.csv'):

            filename = file.name
            print("filename:",filename)
            plot = PhotometrySinglePlot(title=filename)

            if "bkg" in filename:
                _ = plot.addData(file, labelPrefix=f"None", marker="x", color="blue")
            else:
                _ = plot.addData(file, labelPrefix=f"None", marker="D", color="red")

            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

            #_ = plot.plotScatter(0, "TE", plotError=False, verticalLine=False, verticalLineX=0)
            #plot.save(args.output_dir, f"plotScatter")
            
            _ = plot.plotScatterSingleAxes(verticalLine=False, verticalLineX=0)
            plot.save(args.output_dir, f"{filename}")



