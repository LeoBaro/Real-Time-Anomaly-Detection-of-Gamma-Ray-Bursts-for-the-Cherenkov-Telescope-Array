import os
import numpy as np
from time import time
from math import sqrt
from pathlib import Path
from datetime import datetime 
from functools import partial
from multiprocessing import Pool
from dataclasses import dataclass
from astropy.io import fits

from RTAscience.cfg.Config import Config as RTAscienceConfig

from sagsci.tools.plotting import SkyImage
from sagsci.tools.utils import get_obs_pointing
from sagsci.wrappers.rtaph.photometry import Photometrics, aeff_eval

from rtapipe.lib.utils.misc import dotdict
from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils

@dataclass
class SimulationParams:
    runid   : str
    onset   : float
    emin    : float
    emax    : float
    tmin    : int
    tobs    : int
    offset  : float
    irf     : str
    roi     : float
    caldb   : str

    @staticmethod
    def get_from_rta_science_yaml_config(config_path):
        config = RTAscienceConfig(Path(config_path))
        return SimulationParams(
            runid   = config.get('runid'),
            onset   = config.get('onset'),
            emin    = config.get('emin'),
            emax    = config.get('emax'),
            delay    = config.get('delay'),
            tobs    = config.get('tobs'),
            offset  = config.get('offset'),
            irf     = config.get('irf'),
            roi     = config.get('roi'),
            caldb   = config.get('caldb')
        )
"""
class MetaRegion(type):
    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith("__"):
                yield attr
"""
@dataclass
class Region:
    offset         : float
    ra             : float
    dec            : float
    rad            : float
    effective_area : dict
    is_target      : bool
    def get_dict(self):
        return {
            "ra"             : self.ra,
            "dec"            : self.dec,
            "rad"            : self.rad,
        }

@dataclass
class RegionsConfig:

    def __init__(self, regions_radius, max_offset):
        self.regions_radius = regions_radius
        self.max_offset = max_offset
        self.rings = {}
        self.flatten_rings = []

        # Target region is separated from the rest of the regions
        self.target_region = None
        self.target_region_offset = None
        self.target_region_effective_area = {}

    def is_overlapping(self, r1, r2):
        d = sqrt((r1.ra - r2.ra)**2 + (r1.dec - r2.dec)**2)
        return d < r1.rad + r2.rad

    def compute_rings_regions(self, pointing, add_target_region=None, remove_overlapping_regions_with_target=False):

        if pointing is None:
            raise ValueError("Pointing is None")

        offset = 2*self.regions_radius
        while offset <= self.max_offset:
            starting_region = self._add_offset_to_region(pointing, offset)
            ring_regions = Photometrics.find_off_regions('reflection', starting_region.get_dict(), pointing, self.regions_radius)
            self.rings[offset] = [starting_region] + [Region(offset, rr["ra"], rr["dec"], self.regions_radius, {}, False) for rr in ring_regions]
            offset += 2*self.regions_radius
            offset = round(offset, 2)

        if add_target_region:
            # TODO: compute OFFSET between target and pointing.
            # FOR NOW, we assume that the offset is always fixed as 0.5 deg
            target_region_offset = 0.5
            self.target_region = Region(target_region_offset, add_target_region["ra"], add_target_region["dec"], self.regions_radius, {}, True)
            self.target_region_offset = target_region_offset

            if remove_overlapping_regions_with_target:
                offsets = list(self.rings.keys())
                for ii, offset in enumerate(offsets):
                    if offset < target_region_offset:
                        continue
                    else:
                        break
                # overlaps with ii and ii-1 
                self.rings[offsets[ii-1]] = [x for x in self.rings[offsets[ii-1]] if not self.is_overlapping(x, self.target_region)]
                self.rings[offsets[ii]] = [x for x in self.rings[offsets[ii]] if not self.is_overlapping(x, self.target_region)]

    # add energy bins!
    def compute_effective_area(self, irf, pointing, energy_windows):
            args = dotdict({
                'emin': None,
                'emax': None,
                'pixel_size': 0.05,
                "power_law_index" : -2.1,
                "irf_file" : irf
            })
            for e_window in energy_windows:

                args.emin = e_window[0]
                args.emax = e_window[1]

                if self.target_region is not None:
                    self.target_region_effective_area[e_window] = aeff_eval(args, self.target_region.get_dict(), pointing)
                
                for regions in self.rings.values():
                    effective_area = aeff_eval(args, regions[0].get_dict(), pointing)
                    for r in regions:
                        r.effective_area[e_window] = effective_area

    def _add_offset_to_region(self, pointing, offset, is_target=False):
        return Region(offset, pointing["ra"], pointing["dec"] + offset, self.regions_radius, {}, is_target)

    def get_flatten_configuration(self):
        if len(self.flatten_rings) > 0:
            return self.flatten_rings
        for regions in self.rings.values():
            self.flatten_rings += regions
        return self.flatten_rings






class OnlinePhotometry:
    """
    export CTOOLS=/data01/homes/baroncelli/.conda/envs/bphd
    """
    def __init__(self, simulation_params : SimulationParams, integration_time, tsl, number_of_energy_bins):

        if "DATA" not in os.environ:
            raise EnvironmentError("Please, export $DATA")
        
        if "CTOOLS" not in os.environ:
            raise EnvironmentError("Please, export $CTOOLS")

        self.simulation_params = simulation_params
        self.integration_time = integration_time
        self.tsl = tsl
        self.number_of_energy_bins = number_of_energy_bins
        
        self.time_windows = OnlinePhotometry.get_time_windows(simulation_params.tobs, self.integration_time, self.tsl)
        self.energy_windows = PhotometryUtils.getLogWindows(simulation_params.emin, simulation_params.emax, self.number_of_energy_bins)
        self.regions_config = None

    @staticmethod
    def get_target(fits_file):
        with fits.open(fits_file) as hdul:
            ra = abs(hdul[0].header['RA'])
            dec = hdul[0].header['DEC']
        return {"ra": ra, "dec": dec}

    @staticmethod
    def get_time_windows(tobs, integration_time, max_points=None):
        if max_points is not None:
            new_t_obs = max_points * integration_time
            if new_t_obs > tobs:
                raise ValueError(f"max_points * integrationtime ({tobs}) > tobs ({tobs})")
            tobs = new_t_obs
        return PhotometryUtils.getLinearWindows(0, tobs , int(integration_time), int(integration_time))


    def preconfigure_regions(self, regions_radius, max_offset, example_fits, add_target_region=False, remove_overlapping_regions_with_target=False, compute_effective_area_for_normalization=True):
        """
        Compute the regions configuration and the effective area for each region.
        """
        self.regions_config = RegionsConfig(regions_radius, max_offset)
        pointing = get_obs_pointing(example_fits)

        target = None
        if add_target_region:
            template =  Path(os.environ["DATA"]).joinpath("templates",f"{self.simulation_params.runid}.fits")
            target = OnlinePhotometry.get_target(template)

        self.regions_config.compute_rings_regions(pointing, add_target_region=target, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)

        if compute_effective_area_for_normalization:
            irf = Path(os.environ['CTOOLS']).joinpath("share","caldb","data","cta",self.simulation_params.caldb,"bcf",self.simulation_params.irf)
            irf = irf.joinpath(os.listdir(irf).pop())
            self.regions_config.compute_effective_area(irf, pointing, self.energy_windows)
        
        return self.regions_config

    def generate_skymap_with_regions(self, pht_list, output_dir):
        plot = SkyImage()

        if self.regions_config is None:
            raise ValueError("You must call preconfigure_regions before")

        template =  Path(os.environ["DATA"]).joinpath("templates",f"{self.simulation_params.runid}.fits")
        target = OnlinePhotometry.get_target(template)
        plot.set_target(ra=target["ra"], dec=target["dec"])
        
        pointing = get_obs_pointing(pht_list)
        plot.set_pointing(ra=pointing["ra"], dec=pointing["dec"])
        # add timestamp to filename
        out = Path(pht_list.replace('.fits', f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
        out = Path(output_dir).joinpath(out.name)

        regions = [region.get_dict() for region in self.regions_config.get_flatten_configuration()]

        print(f"Producing.. {out}")
        plot.counts_map_with_regions(
                pht_list, 
                regions, 
                trange=None, 
                erange=[self.simulation_params.emin, self.simulation_params.emax],
                roi=self.simulation_params.roi,
                name=out, 
                title="Skymap"
        )
        del plot
    
    

    def integrate(self, pht_list, normalize=True, threads=10, with_metadata=False, regions_radius=None, max_offset=None, example_fits=None, add_target_region=None, remove_overlapping_regions_with_target=None):
        t = time()
        phm = Photometrics({ 'events_filename': pht_list })

        if self.regions_config is None:
            self.preconfigure_regions(regions_radius, max_offset, example_fits, add_target_region, remove_overlapping_regions_with_target, compute_effective_area_for_normalization=normalize)

        func = partial(self.extract_sequence, phm, self.regions_config.regions_radius, self.time_windows, self.energy_windows, normalize)

        regions = self.regions_config.get_flatten_configuration()
        with Pool(threads) as p:
            output = p.map(func, regions)
        
        output = np.asarray(output)

        data = output[:,0,:,:]
        data_err = output[:,1,:,:]

        if with_metadata:
            metadata = {
                "number_of_regions": len(regions),
                "normalize": normalize,
                "elapsed_time" : time() - t
            }
            return data, data_err, metadata

        return data, data_err, None

       
    def extract_sequence(self, phm, region_radius, t_windows, e_windows, normalize, region: Region):
        data = []
        data_err = []
        region = region
        aeff_area = region.effective_area
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
        return np.asarray(data), np.asarray(data_err)                  
