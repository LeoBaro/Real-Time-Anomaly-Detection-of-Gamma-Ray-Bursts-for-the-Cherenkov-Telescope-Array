import re
import os.path 
import argparse
import random
import string
from pathlib import Path
import ntpath
from math import sqrt
from numpy import square

from tqdm import tqdm

from astro.lib.photometry import Photometrics

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
    
    def _integrate(self, integration_on, phm, windows, target_window, output_file, region, photometry_params, sim_params):
        
        if not target_window:
            if integration_on == "t":
                target_window = (sim_params["e_window_start"], sim_params["e_window_stop"]) 
            else:
                target_window = (sim_params["t_window_start"], sim_params["t_window_stop"]) 
        
        # print(f"\n\nIntegration on {integration_on}, len(windows)={len(windows)} [{windows[0]}-{windows[-1]}], target_window={target_window}")   

        total = 0
        with open(output_file, "w") as of:
            of.write("VALMIN,VALMAX,VALCENTER,COUNTS,ERROR\n")    
            total = 0
            ## PARALLELIZE ME! PLEASE :(
            for w in tqdm(windows,disable=True):
                if integration_on == "t":
                    region_count = phm.region_counter(region, photometry_params["region_radius"], tmin=w[0], tmax=w[1], emin=target_window[0], emax=target_window[1])
                    # print(f"region_count: {region_count}, rr: {photometry_params['region_radius']}, tmin={w[0]} tmax={w[1]} emin={target_window[0]} emax={target_window[1]}")
                    # input("..")
                elif integration_on == "e":
                    region_count = phm.region_counter(region, photometry_params["region_radius"], tmin=target_window[0], tmax=target_window[1], emin=w[0], emax=w[1])
                total += region_count
                of.write(f"{round(w[0],4)},{round(w[1], 4)},{round( (w[1]+w[0])/2, 4)},{region_count},{round(sqrt(region_count), 4)}\n")
        ###########################=> print(f"Total counts = {total}, Produced: {output_file}")
        # print(f"Produced {output_file}")
        return output_file

    def getOutFilename(self, photometry_params, sim_params, suffix=""):
        
        output_dir = os.path.join(sim_params["output_dir"], "csv")   
        
        if sim_params["onset"] == 0:
            simtype = sim_params['simtype']
        else:
            simtype = sim_params['simtype']+"_onset"+str(sim_params["onset"])

        input_file_basename = ntpath.basename(sim_params["input_file"])
        input_file_basename = re.sub('\.fits$', '', input_file_basename)        
        output_filename = f"{input_file_basename}_simtype_{simtype}_{suffix}.csv"

        return os.path.join(output_dir, output_filename)    

    def integrate(self, photometry_params, sim_params, integration):
        # print(f"\ngenerate() has been called! integration={integration}")

        runid = sim_params["runid"]  
        simtype = sim_params["simtype"]  
        region_radius = photometry_params["region_radius"]
        
        outdir = Path(sim_params["output_dir"])
        outdir.joinpath("csv").mkdir(exist_ok=True, parents=True)

        time_windows = self.getWindows(sim_params["t_window_start"], sim_params["t_window_stop"], photometry_params["t_window_size"], photometry_params["t_window_step"])
        energy_windows = self.getWindows(sim_params["e_window_start"], sim_params["e_window_stop"], photometry_params["e_window_size"], photometry_params["e_window_step"])

        input_file = sim_params["input_file"]

        phm = Photometrics({ 'events_filename': input_file })
        region = {
            'ra': photometry_params["pointing"][0],
            'dec': photometry_params["pointing"][1],
        }        
        #print(f"region: {region}")
        #print(f"File: {input_file}, pointing: {region}")

        # pool = multiprocessing.Pool(4)
        # out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
        if integration == "t":
            output_file = self._integrate("t", phm, time_windows, None, self.getOutFilename(photometry_params, sim_params, suffix="t"), region, photometry_params, sim_params)
            return [output_file]
       
        
        elif integration == "e":
            output_file = self._integrate("e", phm, energy_windows, None, self.getOutFilename(photometry_params, sim_params, suffix="e"), region, photometry_params, sim_params)
            return [output_file]

        elif integration == "te":
            output_files = []
            for id, time_window in tqdm(enumerate(time_windows),disable=True):
                output_file = self._integrate("e", phm, energy_windows, time_window, self.getOutFilename(photometry_params, sim_params, suffix=f"e_time_window_{id}"), region, photometry_params, sim_params)
                output_files.append(output_file)
            return output_files

        elif integration == "et":
            output_files = []
            for id, energy_window in tqdm(enumerate(energy_windows),disable=True):
                output_file = self._integrate("t", phm, time_windows, energy_window, self.getOutFilename(photometry_params, sim_params, suffix=f"f_energy_window_{round(energy_window[0], 2)}_{round(energy_window[1], 2)}"), region, photometry_params, sim_params)
                output_files.append(output_file)
            return output_files



if __name__ == '__main__':
    """
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
    """
        