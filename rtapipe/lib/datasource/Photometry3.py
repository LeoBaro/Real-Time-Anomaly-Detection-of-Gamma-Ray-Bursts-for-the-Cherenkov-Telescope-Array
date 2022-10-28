import pickle
from time import time
from math import sqrt
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from sagsci.tools.utils import *
from sagsci.wrappers.rtaph.photometry import Photometrics, aeff_eval

from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class OnlinePhotometry(Photometry2):
    """
    This class only support TIME-ENERGY integration

    export CTOOLS=/data01/homes/baroncelli/.conda/envs/bphd
    """
    def __init__(self, configPath):
        super().__init__(configPath, None, "/tmp")


    def get_time_windows(self, integration_time, max_points):
        if max_points is not None:
            new_t_obs = max_points * integration_time
            if new_t_obs > self.cfg.get("tobs"):
                raise ValueError(f"max_points * integrationtime ({tobs}) > tobs ({tobs})")
            tobs = new_t_obs
        return PhotometryUtils.getLinearWindows(0, tobs , int(integration_time), int(integration_time))


    def get_energy_windows(self, number_of_energy_bins):
        return PhotometryUtils.getLogWindows(self.cfg.get("emin"), self.cfg.get("emax"), number_of_energy_bins)


    def get_aeff_eval_config(self, e_windows):
        aeff_eval_config = {}
        for ew in e_windows:
            aeff_eval_config[ew] = dotdict({
                'emin': ew[0],
                'emax': ew[1],
                'pixel_size': 0.05,
                "power_law_index" : -2.1,
                "irf_file" : self.irfFile
            })
        return aeff_eval_config
        
    def get_starting_region(self, offset, region_radius):
        return {
            'ra': self.sourcePosition[0] + offset, 
            'dec': self.sourcePosition[1], 
            'rad': region_radius
        }


    def compute_region(self, e_windows, region_radius):

        aeff_eval_config = self.get_aeff_eval_config(e_windows)
        offset = self.cfg.get("offset")
        pointing_dict = {
            'ra' : self.sourcePosition[0],
            'dec' : self.sourcePosition[1]
        }
        regions_dict = {}
        target_region = self.get_starting_region(offset, region_radius)
        # Compute the effective response of the region (given offset and energy bins)
        regions_dict[offset] = {
            "region_eff_resp" : {},
            "regions": []
        }
        for ew in e_windows:
            regions_dict[offset]["region_eff_resp"][ew] = aeff_eval(aeff_eval_config[ew], target_region, pointing_dict)
        regions_dict[offset]["regions"].append(target_region) 
        return regions_dict

        

    def compute_reflected_regions(self, max_offset, e_windows, region_radius):

        aeff_eval_config = self.get_aeff_eval_config(e_windows)
        offset = self.cfg.get("offset")
        pointing_dict = {
            'ra' : self.sourcePosition[0],
            'dec' : self.sourcePosition[1]
        }

        regions_dict = {}

        # define the rings in FOV
        while offset <= max_offset:
            
            # define the starting (aka target not necessary source) region
            target_region = self.get_starting_region(offset, region_radius)

            # Compute the effective response of the region (given offset and energy bins)
            regions_dict[offset] = {
                "region_eff_resp" : {},
                "regions": []
            }
            for ew in e_windows:
                regions_dict[offset]["region_eff_resp"][ew] = aeff_eval(aeff_eval_config[ew], target_region, pointing_dict)
           
            # get ring of regions from starting position
            ring_regions = Photometrics.find_off_regions('reflection', target_region, pointing_dict, region_radius)

            # add starting position region to dictionary 
            ring_regions += [target_region]

            # for each region in ring count events and get flux
            regions_dict[offset]["regions"] = ring_regions

            # increment offset to get next offset ring
            offset += region_radius*2
        
        return regions_dict

    def flat_region_dict(self, regions_dict, max_offset, e_windows):
        # change the structure of the dictionary: list of tuple of region and aeff
        # [( 
        #    {'ra': 31.68167110956259, 'dec': -52.8402349288972, 'rad': 0.2}, {(0.04, 0.117): 584733789.0746386, (0.117, 0.342): 1568499319.2974534, (0.342, 1.0): 3159516976.6579475} 
        # )]
        t = time()
        flattened_regions = []
        for offset in regions_dict:
            for region in regions_dict[offset]["regions"]:
                flattened_regions.append((region, regions_dict[offset]["region_eff_resp"]))
        print(f"Time to change the structure of the config dictionary: {time() - t}")
        return flattened_regions


    def create_photometry_configuration(self, region_radius, number_of_energy_bins, max_points, max_offset=2, reflection=True):
        t = time()
        e_windows = self.get_energy_windows(number_of_energy_bins)
        regions_dict = self.compute_region(e_windows, region_radius)
        if reflection:
            regions_dict = self.compute_reflected_regions(max_offset, e_windows, region_radius)
            regions_dict = self.flat_region_dict(regions_dict, max_offset, e_windows)
        print(f"Time to compute regions: {time() - t}")
        return regions_dict


    def integrate(self, pht_list, regions_dict, region_radius, integration_time, number_of_energy_bins, max_points, normalize=True, threads=10):
        t = time()
        phm = Photometrics({ 'events_filename': pht_list })
        print(f"Time to load Photometry class: {time() - t}")

        t_windows = self.get_time_windows(integration_time, max_points)
        e_windows = self.get_energy_windows(number_of_energy_bins)

        func = partial(self.extract_sequence, phm, t_windows, e_windows, region_radius, normalize)

        with Pool(threads) as p:
            output = p.map(func, regions_dict)
        
        output = np.asarray(output)
        print(output.shape)

        data = output[:,0,:,:]
        data_err = output[:,1,:,:]

        return data, data_err

       
    def extract_sequence(self, phm, t_windows, e_windows, region_radius, normalize, region_config):
        t = time()
        data = []
        data_err = []
        region = region_config[0]
        aeff_area = region_config[1]
        livetime = t_windows[0][1] - t_windows[0][0]
        for twin in t_windows:
            counts_t = []
            counts_t_err = []
            for ewin in e_windows:
                counts = phm.region_counter(region, region_radius, tmin=twin[0], tmax=twin[1], emin=ewin[0], emax=ewin[1])
                if normalize:                    
                    counts = counts / aeff_area[ewin] / livetime
                    error = 0 # TODO: compute error on FLUX
                else:
                    error = sqrt(counts)                 
                counts_t.append(counts)
                counts_t_err.append(error)   
            data.append(counts_t)
            data_err.append(counts_t_err)
        print(f"Time to extract sequence: {time() - t}")
        return np.asarray(data), np.asarray(data_err)                  
