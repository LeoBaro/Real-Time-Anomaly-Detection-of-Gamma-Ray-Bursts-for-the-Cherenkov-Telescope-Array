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

plt.rcParams.update({'font.size': 10})

class PhotometryPlot(ABC):
    
    inch_x = 10
    inch_y = 6
    colors = ["b","g","r","c"]
    ccount = 0
    
    def __init__(self):
        self.outPlot = None


    def getData(self, photometry_csv_file, integration):
        df = pd.read_csv(photometry_csv_file)
        if integration == "t" or integration=="e": 
            return {
                "x": df['VALCENTER'],
                "y": df['COUNTS'],
                "err": df['ERROR']
            }
        if integration == "t,e": 
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
        
        _ = self.FSX.scatter(data["x"], data["y"], label=label_on+"_"+str(args[label_on]), color=PhotometryPlot.colors[self.ccount])
        _ = self.FSX.errorbar(data["x"], data["y"], yerr=data["err"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
        _ = self.FSX.axvline(x=args["onset"], color="red", linestyle="--")
        self.ccount += 1

        forbidden = self.getForbidden() + ["label_on"] 
        ok = ["pointing", "regios_radius", "t_window_start", "t_window_size", "t_window_step", "onset"]

        title = f'Aperture Photometry Counts: \n {[f"{key}={value} " for key, value in args.items() if key not in forbidden and key in ok]}'
        self.FSX.set_title(title)
        self.FSX.set_ylabel('Windowed Counts')
        self.FSX.set_xlabel(f'Window center')
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
        self.FMi, self.FMj, self.FMc = 0, 0, 0  
        self.FM.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)


    def addData(self, photometry_csv_file, args, label_on, integration):
        data = super().getData(photometry_csv_file, integration)

        axis = self.FMX[self.FMi][self.FMj]

        _ = axis.scatter(data["x"], data["y"], label=label_on+"_"+str(args[label_on]), color=PhotometryPlot.colors[self.ccount])
        _ = axis.errorbar(data["x"], data["y"], yerr=data["err"], fmt="o", color=PhotometryPlot.colors[self.ccount])
        _ = axis.axvline(x=args["onset"], color="red", linestyle="--")
        self.ccount += 1

        print(f"FMi: {self.FMi} FMj: {self.FMj}")
        if self.FMc % 2 == 0:
            self.FMi = (self.FMi + 1) % 2
        else:
            self.FMj = (self.FMj + 1) % 2
        print(f"FMi: {self.FMi} FMj: {self.FMj}")
        self.FMc+=1

        forbidden = self.getForbidden() + ["label_on"] 
        ok = ["pointing", "regios_radius", "t_window_start", "t_window_size", "t_window_step", "onset"]

        title = f'Aperture Photometry Counts: \n {[f"{key}={value} " for key, value in args.items() if key not in forbidden and key in ok ]}'
        self.FM.suptitle(title)
        axis.set_title(f"{label_on}: {args[label_on]}")
        axis.set_ylabel('Windowed Counts')
        axis.set_xlabel(f'Window center')
                
        self.FM.tight_layout()

    def save(self, outfileName):
        outfileName = outfileName+'.png'
        self.FM.savefig(outfileName)
        print("Produced: ", outfileName)

    def show(self):
        plt.show()



class Photometry:

    def getWindows(self, window_start, window_stop, window_size, window_step):

        print(f"window_start {window_start}, window_stop {window_stop}, window_size {window_size}, window_step {window_step}")
        if window_size == 0:
            raise ValueError("window-size must be greater than zero.")

        _windows = []

        t_start = window_start
        t_stop = window_stop
        w_start = 0
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


    def generate(self, args, integration):

        runid = args["runid"]  
        simtype = args["simtype"]  
        region_radius = args["region_radius"]

        time_windows = self.getWindows(args["t_window_start"], args["t_window_stop"], args["t_window_size"], args["t_window_step"])
        energy_windows = self.getWindows(args["e_window_start"], args["e_window_stop"], args["e_window_size"], args["e_window_step"])

        #print(f"time_windows: {time_windows}")
        print(f"energy_windows: {energy_windows}")
        
        
        input_file = None
        if simtype == 'grb':
            input_file = os.path.join(args["obs_dir"], runid, "ebl000001.fits")
            output_dir = os.path.join(args["obs_dir"], runid, "ap")
            output_file = os.path.join(output_dir, f"ebl000001_ap_mode")   
        elif simtype == 'bkg': # bkg
            input_file = os.path.join(args["obs_dir"], "backgrounds", "bkg000001.fits")
            output_dir = os.path.join(args["obs_dir"], "backgrounds", "ap")
            output_file = os.path.join(output_dir, f"bkg000001_ap_mode")

        try: 
            os.mkdir(output_dir) 
        except OSError as error: 
            pass 


        #if args["mode == "cumulative":
        #    output_file += f"_{args["mode}_wsize_{args["window_size}_rad_{args["region_radius}.csv"
        #elif args["mode == "windowed":
        output_file += f"_windowed-mode_integration_{integration}_"
        output_file += self.get_random_string(15)+".csv"

        phm = Photometrics({ 'events_filename': input_file })
        region = {
            'ra': args["pointing"][0],
            'dec': args["pointing"][1],
        }

        # print('File: ', input_file)
        print('\nRegion center ', region, 'with radius', args["region_radius"], 'deg')
        
        total = 0
        if os.path.isfile(output_file) and args["override"] == 0:
            print(f"File {output_file} already exists!")
        else:


            if integration == "t":
                with open(output_file, "w") as of:
                    of.write("VALMIN,VALMAX,VALCENTER,COUNTS,ERROR\n")    
                    for time_window in time_windows:
                        region_count = phm.region_counter(region, args["region_radius"], tmin=time_window[0], tmax=time_window[1])
                        total += region_count
                        # print(f'tmin {time_window[0]} tmax {time_window[1]} -> counts: {region_count}')
                        of.write(f"{time_window[0]},{time_window[1]},{(time_window[1]+time_window[0])/2},{region_count},{sqrt(region_count)}\n")
                print("Produced: ",output_file)
            
            elif integration == "e":
                with open(output_file, "w") as of:
                    of.write("VALMIN,VALMAX,VALCENTER,COUNTS,ERROR\n")    
                    for energy_window in energy_windows:
                        region_count = phm.region_counter(region, args["region_radius"], emin=energy_window[0], emax=energy_window[1])
                        total += region_count
                        of.write(f"{energy_window[0]},{energy_window[1]},{(energy_window[1]+energy_window[0])/2},{region_count},{sqrt(region_count)}\n")
                print("Produced: ",output_file)

            elif integration == "t,e":
                with open(output_file, "w") as of:
                    of.write("TMIN,TMAX,TMID,EMIN,EMAX,EMID,COUNTS,ERROR\n")    
                    for time_window in time_windows:
                        for energy_window in energy_windows:
                            region_count = phm.region_counter(region, args["region_radius"], tmin=time_window[0], tmax=time_window[1], emin=energy_window[0], emax=energy_window[1])
                            total += region_count
                            # print(f'tmin {time_window[0]} tmax {time_window[1]} -> counts: {region_count}')
                            of.write(f"{time_window[0]},{time_window[1]},{(time_window[1]+time_window[0])/2},{energy_window[0]},{energy_window[1]},{(energy_window[1]+energy_window[0])/2},{region_count},{sqrt(region_count)}\n")
                print("Produced: ",output_file)

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