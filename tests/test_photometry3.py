import numpy as np
from time import time
import tensorflow as tf
from pathlib import Path
from rtapipe.lib.datasource.Photometry3 import OnlinePhotometry

class TestPhotometry3:

    def test_integrate(self):
        
        data_dir = "/data01/homes/baroncelli/phd/tests/test_data/fits-prod5/bkg_only"
        config_file = Path(data_dir).joinpath("config.yaml")
        photon_list = Path(data_dir).joinpath("runid_notemplate_trial_0001024120_simtype_bkg_onset_0_delay_0_offset_0.5.fits")
        
        pht = OnlinePhotometry(config_file, data_dir)

        region_radius = 0.2
        integration_time = 5
        number_of_energy_bins = 3
        max_points = 5
        reflection = True
        normalize = True
        threads = 10

        s = time()
        data, data_err = pht.integrate(photon_list, region_radius, integration_time, number_of_energy_bins, max_points, reflection=reflection, normalize=normalize, threads=threads)
        print(f"Time to integrate: {time() - s} sec")

        assert data.shape == (56, 5, 3)
        print(data[0])

        threads = 1
        s = time()
        data, data_err = pht.integrate(photon_list, region_radius, integration_time, number_of_energy_bins, max_points, reflection=reflection, normalize=normalize, threads=threads)
        print(f"Time to integrate: {time() - s} sec")


        normalize = False
        threads = 10
        data, data_err = pht.integrate(photon_list, region_radius, integration_time, number_of_energy_bins, max_points, reflection=reflection, normalize=normalize, threads=threads)
        print(data[0])
