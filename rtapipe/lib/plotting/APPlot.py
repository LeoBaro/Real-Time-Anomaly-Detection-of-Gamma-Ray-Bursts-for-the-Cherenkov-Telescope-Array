import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation

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

    def plot_from_numpy(self, data, params, labels=[]):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=self.pc.fig_size)
        self.pc.colors
        for i in range(data.shape[1]):
            if len(labels) > 0:
                label = labels[i]
            else:
                label = f"Feature {i}"   
            error_bar_kwargs = {
                "ls": "none",
                "marker": 'o',
                "color": self.pc.colors[i], 
                "ecolor": self.pc.colors[i]
            }     
            #self.ax.errorbar(range(data.shape[0]), data[:,i], label=label, **self.pc.error_kw, **error_bar_kwargs)
            self.ax.plot(data[:,i], color=self.pc.colors[i], marker='o', markersize=6, linestyle='dashed', label=label)

        if params["onset"] > 0:
            pivot_idx = params["onset"] // params["itime"]
            self.ax.axvline(x = pivot_idx, color = "purple", linestyle = 'dashed', label = "GRB start")

        self.set_layout(params)
    
    def plot_sliding_window(self, data, tsl, stride, params, labels=[]):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=self.pc.fig_size)
        self.ax = plt.axes(xlim=(0, data.shape[0]), ylim=(0, 1))
        lines = []
        for i in range(data.shape[1]):
            label = f"Feature {i}" 
            if len(labels) > 0:
                label = labels[i]
                line, = self.ax.plot([], [], color=self.pc.colors[i], marker='o', markersize=6, linestyle='dashed', label=label)
                lines.append(
                    line
                )
        self.set_layout(params)

        def init():
            for i in range(data.shape[1]):
                lines[i].set_data([], [])
            return lines    

        def animate(i):
            print(i)
            if i > data.shape[0]-tsl:
                anim.event_source.stop()
            else:
                x = list(range(i,i+tsl))
                #print(x)
                for j in range(data.shape[1]):
                    y = data[i:i+tsl,j]
                    #print(y)
                    lines[j].set_data(x,y)

            return lines
        
        anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=data.shape[0]-tsl, interval=500, blit=True, repeat=False)
        anim.save('basic_animation.gif', fps=30)


    def plot(self, csv_file_path, params, start=0, lenght=None):
        df = self.read_data(csv_file_path)
        if lenght is not None and lenght > 0:
            df = df.loc[start:start+lenght,:]

        data_col_names  = [col_name for col_name in df.columns if "COUNTS" in col_name]
        error_col_names = [col_name for col_name in df.columns if "ERROR"  in col_name]

        totc=0
        for dc in data_col_names:
            #print(f"Counts in {dc}={df[dc].sum()}")
            totc += df[dc].sum()
        print(f"Total counts: {totc}")

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

        if params["onset"] > 0:
            pivot_idx = params["onset"] // params["itime"]
            self.ax.axvline(x = pivot_idx, color = "purple", linestyle = 'dashed', label = "GRB start")

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


if __name__=='__main__':
    import numpy as np
    samples = np.random.uniform(0, 1, size=(1, 100, 3))
    ap = APPlot()
    tsl = 5
    stride = 1
    ap.plot_sliding_window(samples[0], tsl, stride, params={"runid": "test", "itime": 5, "itype": "test", "onset": 0, "offset": 0, "maxflux": None, "normalized": False}, labels=["EB_1", "EB_2", "EB_3"])