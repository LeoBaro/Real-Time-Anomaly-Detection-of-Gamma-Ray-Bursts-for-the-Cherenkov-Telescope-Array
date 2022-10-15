import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from rtapipe.lib.plotting.PlotConfig import PlotConfig

class APPlot:

    def __init__(self):
        self.pc = PlotConfig()
        self.fig = None
        self.ax = None

    def read_data(self, csv_file_path):
        df = pd.read_csv(str(csv_file_path))
        if "TMIN" in df.columns:
            df["TCENTER"] = (df["TMAX"] + df['TMIN'])/2
        if "EMIN" in df.columns:
            df["ECENTER"] = (df["EMAX"] + df['EMIN'])/2
        return df

    def set_layout(self, params):
        self.ax.set_xlabel("Time [s]")
        if params["maxflux"] is not None:
            self.ax.set_ylim(0, params["maxflux"])
        if params["normalized"]:
            self.fig.suptitle("Flux in region")
            self.ax.set_ylabel("Flux [ph/cm2/s]")
        else:
            self.fig.suptitle("Counts in region")
            self.ax.set_ylabel("Counts")
        self.ax.set_title(f"Template: {params['runid']}, Integration time: {params['itime']}, Type: {params['itype']}, Onset: {params['onset']}, Offset: {params['offset']}, Region rad: 0.2°")
        self.ax.legend(prop=self.pc.legend_kw, loc='upper left')

    def plot(self, csv_file_path, params, lenght=None):
        df = self.read_data(csv_file_path)
        if lenght is not None and lenght > 0:
            df = df.loc[0:lenght,:]
        print(df.shape)

        data_col_names  = [col_name for col_name in df.columns if "COUNTS" in col_name]
        error_col_names = [col_name for col_name in df.columns if "ERROR"  in col_name]



        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=self.pc.fig_size)

        self.pc.colors

        for i in range(len(data_col_names)):

            energy_range = data_col_names[i].replace("COUNTS_","")
            label = f"Energy {energy_range} TeV"      
            error_bar_kwargs = {
                "ls": "none",
                "marker": 'o',
                "color": self.pc.colors[i], 
                "ecolor": self.pc.colors[i]
            }     
            self.ax.errorbar(df["TCENTER"], df[data_col_names[i]], yerr=df[error_col_names[i]], label=label, **self.pc.error_kw, **error_bar_kwargs)

        self.set_layout(params)


    def save(self, outputDir, outputFilename):
        if self.fig:
            outputDir = Path(outputDir)
            outputDir.mkdir(parents=True, exist_ok=True)
            outputFilePath = outputDir.joinpath(outputFilename).with_suffix(".png")
            self.fig.savefig(str(outputFilePath), dpi=600)
            print(f"Produced: {outputFilePath}")
            return str(outputFilePath)
        print("No plot has been generated yet!")
