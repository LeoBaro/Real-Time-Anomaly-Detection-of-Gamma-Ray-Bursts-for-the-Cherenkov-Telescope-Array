import re
from math import sqrt
from numpy import square
import os.path, argparse
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC
from astro.lib.photometry import Photometrics
import random
import string

from rtapipe.pyscripts.data_utils import DataUtils

plt.rcParams.update({'font.size': 7, 'lines.markersize': 0.5,'legend.markerscale': 5, 'lines.linewidth':0.5, 'lines.linestyle':'--'})

class PhotometryPlot(ABC):
    
    inch_x = 10
    inch_y = 6
    colors = ["b","g","r","c"]
    ccount = 0
    
    def __init__(self):
        self.outPlot = None


    def getData(self, photometry_csv_file, integration):
        df = pd.read_csv(photometry_csv_file)
        # print(df.head(10))
        if integration == "t" or integration=="e": 
            return {
                "x": df['VALCENTER'],
                "y": df['COUNTS'],
                "err": df['ERROR'],
                "valmax": df['VALMAX'],
                "valmin": df['VALMIN']
            }
        if integration == "t+e": 
            return {
                "x1": df['TMID'],
                "x2": df['EMID'],
                "y": df['COUNTS'],
                "err": df['ERROR']
            }            

    def getColor(self):
        col = colors[ccount]
        ccount += 1
        return col

    def getForbidden(self):
        return ["datapath", "simtype", "runid", "obs_dir", "override", "plot"]

    def addData(self, photometry_csv_file, args, label_on, integration):
        pass

    def show(self):
        pass

    def destroy(self):
        plt.close()

class PhotometrySinglePlot(PhotometryPlot):
    
    def __init__(self):
        self.FS, self.FSX  = plt.subplots()
        self.FS.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        self.outputfile = None

    def addData(self, photometry_csv_file, args, label_on, integration):
        data = super().getData(photometry_csv_file, integration)
        # print(f"\naddData called!")

        label_on_string = ""
        for label in label_on:
            if label in args:
                label_on_string += label+"_"+str(args[label])+"_"
            else:
                label_on_string += label

        if integration == "t":
            _ = self.FSX.scatter(data["x"], data["y"], s=0.1, label=label_on_string, color=PhotometryPlot.colors[self.ccount])
            _ = self.FSX.errorbar(data["x"], data["y"], yerr=data["err"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            _ = self.FSX.axvline(x=args["onset"], color="red", linestyle="--")

        elif integration == "e":
            _ = self.FSX.bar(data["x"], data["y"], yerr=data["err"], label=label_on_string , color=PhotometryPlot.colors[self.ccount], alpha=0.1, width=args["e_window_step"])

        
        self.ccount += 1

        forbidden = self.getForbidden() + label_on
        ok = ["pointing", "region_radius", "t_window_start", "t_window_size", "t_window_step", "t_window_stop", "e_window_start", "e_window_size", "e_window_step", "e_window_stop", "onset"]

        kawrgs = [f"{key}: {value} " for key, value in args.items() if key not in forbidden and key in ok]
        title = f'Aperture Photometry Counts: \n'
        title += str(kawrgs[:5]) + "\n" + str(kawrgs[5:])

        self.FSX.set_title(title)
        self.FSX.set_ylabel('Counts')
        self.FSX.set_xlabel(f'Window center (integration: "{integration}")')
        self.FSX.legend(loc="best")

    def show(self):
        plt.show()


    def save(self, outfileName):
        outfileName = outfileName+'.png'
        self.FS.savefig(outfileName)
        print("Produced: ", outfileName)
        
class PhotometrySubPlots(PhotometryPlot):

    def __init__(self):
        self.FM, self.FMX  = plt.subplots(2, 2)
        self.FM.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        self.current_axis_idx = 0
        self.axes = [(0,0), (0,1), (1,0), (1,1)]

    def nextAxis(self):
        self.current_axis_idx = (self.current_axis_idx + 1) % len(self.axes)
        self.ccount = (self.ccount + 1) % len(self.axes)
        x,y = self.axes[self.current_axis_idx]
        return self.FMX[x][y]
    
    def prevAxis(self):
        self.current_axis_idx = (self.current_axis_idx - 1) % len(self.axes)
        self.ccount = (self.ccount - 1) % len(self.axes)
        x,y = self.axes[self.current_axis_idx]
        return self.FMX[x][y]

    def currentAxis(self):
        x,y = self.axes[self.current_axis_idx]     
        return self.FMX[x][y]

    def addData(self, photometry_csv_file, args, label_on, integration, as_baseline=False):
        data = super().getData(photometry_csv_file, integration)
        # print(f"\naddData called! as_baseline={as_baseline} ")

        if as_baseline:
            axis = self.prevAxis()
        else:
            axis = self.currentAxis()
            
        #print("---  current axis: ", self.current_axis_idx)

        label_on_string = ""
        for label in label_on:
            if label in args:
                label_on_string += label+"_"+str(args[label])+"_"
            else:
                label_on_string += label

        if integration == "t":

            if not as_baseline: 
                plotargs = {"color": PhotometryPlot.colors[self.ccount]}
            else:
                plotargs = {"color": "black"}

            if not as_baseline: 
                _ = axis.scatter(data["x"], data["y"], label = label_on_string, **plotargs, s=0.1)
                _ = axis.errorbar(data["x"], data["y"], yerr=data["err"], fmt="o", **plotargs)
                _ = axis.axvline(x=args["onset"], color="red", linestyle="--")
            else:
                _ = axis.scatter(data["x"], data["y"], label = "baseline", **plotargs, s=0.1)
                _ = axis.errorbar(data["x"], data["y"], yerr=data["err"], fmt="o", **plotargs)


        elif integration == "e":
            plotargs = {
                "yerr": data["err"], 
                "label": label_on_string,
                "alpha": 0.2, 
                "width": args["e_window_step"]
            }
            if as_baseline:
                plotargs["linewidth"] = 4
                plotargs["color"] = "black"
            else:
                plotargs["color"] = PhotometryPlot.colors[self.ccount]

            _ = axis.bar(data["x"], data["y"], **plotargs)
            

        forbidden = self.getForbidden() + label_on 
        ok = ["pointing", "region_radius", "t_window_start", "t_window_size", "t_window_step", "t_window_stop", "e_window_start", "e_window_size", "e_window_step", "e_window_stop", "onset"]

        kawrgs = [f"{key}: {value} " for key, value in args.items() if key not in forbidden and key in ok]
        title = f'Aperture Photometry Counts: \n'
        title += str(kawrgs[:5]) + "\n" + str(kawrgs[5:])

        self.FM.suptitle(title)
        axis.set_title(label_on_string)
        axis.set_ylabel('Counts')
        axis.set_xlabel(f'Window center (integration: "{integration}")')

        axis.legend(loc="best")

        self.nextAxis()

        # self.FM.tight_layout()

    def save(self, outfileName):
        outfileName = outfileName+'.png'
        self.FM.savefig(outfileName)
        print("Produced: ", outfileName)

    def show(self):
        plt.show()



class Photometry:

    def getWindows(self, window_start, window_stop, window_size, window_step):

        #print(f"window_start {window_start}, window_stop {window_stop}, window_size {window_size}, window_step {window_step}")
        if window_size == 0:
            raise ValueError("window-size must be greater than zero.")

        _windows = []

        t_stop = window_stop
        w_start = window_start
        while w_start + window_size <= t_stop:
            _windows.append((w_start, w_start + window_size))
            w_start += window_step
        #print("_windows:", _windows)
        return _windows


    def get_random_string(self, length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str


    def integrate(self, photometry_params, sim_params, integration):
        # print(f"\ngenerate() has been called! integration={integration}")
        
        runid = sim_params["runid"]  
        simtype = sim_params["simtype"]  
        region_radius = photometry_params["region_radius"]

        time_windows = self.getWindows(0, sim_params["t_window_stop"], photometry_params["t_window_size"], photometry_params["t_window_step"])
        energy_windows = self.getWindows(sim_params["e_window_start"], sim_params["e_window_stop"], photometry_params["e_window_size"], photometry_params["e_window_step"])

        #print(f"time_windows: {time_windows}")
        #print(f"energy_windows: {energy_windows}")
        
        
        input_file = None
        if simtype == 'grb':
            input_file = os.path.join(sim_params["obs_dir"], runid, "ebl000001.fits")
            output_dir = os.path.join(sim_params["obs_dir"], runid, "ap")
            output_file = os.path.join(output_dir, f"ebl000001_ap_mode")   
        elif simtype == 'bkg': # bkg
            input_file = os.path.join(sim_params["obs_dir"], "backgrounds", "bkg000001.fits")
            output_dir = os.path.join(sim_params["obs_dir"], "backgrounds", "ap")
            output_file = os.path.join(output_dir, f"bkg000001_ap_mode")

        try: 
            os.mkdir(output_dir) 
        except OSError as error: 
            pass 


        output_file += f"_windowed-mode_integration_{integration}_"
        output_file += self.get_random_string(15)+".csv"

        phm = Photometrics({ 'events_filename': input_file })
        region = {
            'ra': photometry_params["pointing"][0],
            'dec': photometry_params["pointing"][1],
        }

        # print('File: ', input_file)
        # print('\nRegion center ', region, 'with radius', photometry_params["region_radius"], 'deg')
        
        total = 0
        if os.path.isfile(output_file) and photometry_params["override"] == 0:
            print(f"File {output_file} already exists!")
        else:


            if integration == "t":
                windows = time_windows
                with open(output_file, "w") as of:
                    of.write("VALMIN,VALMAX,VALCENTER,COUNTS,ERROR\n")    
                    for window in windows:
                        region_count = phm.region_counter(region, photometry_params["region_radius"], tmin=window[0], tmax=window[1], emin=sim_params["e_window_start"], emax=sim_params["e_window_stop"])
                        total += region_count
                        # print(f'tmin {window[0]} tmax {window[1]} -> counts: {region_count}')
                        window_center = (window[1]+window[0])/2
                        of.write(f"{window[0]},{window[1]},{window_center},{region_count},{sqrt(region_count)}\n")

            
            
            elif integration == "e":
                windows = energy_windows
                with open(output_file, "w") as of:
                    of.write("VALMIN,VALMAX,VALCENTER,COUNTS,ERROR\n")    
                    for window in windows:
                        region_count = phm.region_counter(region, photometry_params["region_radius"], tmin=sim_params["t_window_start"], tmax=sim_params["t_window_stop"], emin=window[0], emax=window[1])
                        total += round(region_count, 2)
                        # print(f'tmin {window[0]} tmax {window[1]} -> counts: {region_count}')
                        window_center = (window[1]+window[0])/2
                        of.write(f"{window[0]},{window[1]},{window_center},{region_count},{sqrt(region_count)}\n")
            

            print("Produced: ",output_file)

            """
            elif integration == "t,e":
                with open(output_file, "w") as of:
                    of.write("TMIN,TMAX,TMID,EMIN,EMAX,EMID,COUNTS,ERROR\n")    
                    for time_window in time_windows:
                        for energy_window in energy_windows:
                            region_count = phm.region_counter(region, photometry_params["region_radius"], tmin=time_window[0], tmax=time_window[1], emin=energy_window[0], emax=energy_window[1])
                            total += region_count
                            # print(f'tmin {time_window[0]} tmax {time_window[1]} -> counts: {region_count}')
                            of.write(f"{time_window[0]},{time_window[1]},{(time_window[1]+time_window[0])/2},{energy_window[0]},{energy_window[1]},{(energy_window[1]+energy_window[0])/2},{region_count},{sqrt(region_count)}\n")
                print("Produced: ",output_file)
            """
        return output_file
                


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