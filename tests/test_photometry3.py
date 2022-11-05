import os
import pytest
from pathlib import Path
from rtapipe.lib.datasource.Photometry3 import RegionsConfig, SimulationParams, Region, OnlinePhotometry

class TestPhotometry3:

    def test_region_config_rings_regions_without_target(self):

        rc = RegionsConfig(0.2, 2.5)

        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0})
        
        assert len(rc.rings.keys()) == 6 
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0, 2.4]
        assert len(rc.rings[0.4]) == 6
        assert len(rc.rings[0.8]) == 12
        assert len(rc.rings[1.2]) == 18
        assert len(rc.rings[1.6]) == 24
        assert len(rc.rings[2.0]) == 30
        assert len(rc.rings[2.4]) == 36
        
        os.environ["CTOOLS"] = "/data01/homes/baroncelli/.conda/envs/bphd"
        irf = Path(os.environ['CTOOLS']).joinpath("share","caldb","data","cta","prod5-v0.1","bcf","North_z40_5h_LST")
        irf = irf.joinpath(os.listdir(irf).pop())

        regions = rc.get_flatten_configuration()
        assert len(regions) == 6+12+18+24+30+36

        rc.compute_effective_area(irf, {"ra":20.0, "dec": -50.0}, 0.04, 1)        
        for regions in rc.rings.values():
            for region in regions:
                assert region.effective_area > 0.0


    def test_region_config_rings_regions_with_target(self):

        rc = RegionsConfig(0.2, 2.5)

        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0}, add_target_region={"ra":19.5, "dec": -50.0})
        
        assert len(rc.rings.keys()) == 7
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 0.5]
        assert len(rc.rings[0.4]) == 6
        assert len(rc.rings[0.5]) == 1
        assert len(rc.rings[0.8]) == 12
        assert len(rc.rings[1.2]) == 18
        assert len(rc.rings[1.6]) == 24
        assert len(rc.rings[2.0]) == 30
        assert len(rc.rings[2.4]) == 36        

        regions = rc.get_flatten_configuration()
        assert len(regions) == 6+12+18+24+30+36+1

    def test_region_config_rings_regions_with_target_no_overlaps(self):

        rc = RegionsConfig(0.2, 2.5)
        remove_overlapping_regions_with_target = True
        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0}, add_target_region={"ra":19.5, "dec": -50.0}, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)
        
        assert len(rc.rings.keys()) == 7
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 0.5]
        assert len(rc.rings[0.4]) == 5
        assert len(rc.rings[0.5]) == 1
        assert len(rc.rings[0.8]) == 11
        assert len(rc.rings[1.2]) == 18
        assert len(rc.rings[1.6]) == 24
        assert len(rc.rings[2.0]) == 30
        assert len(rc.rings[2.4]) == 36        

        regions = rc.get_flatten_configuration()
        assert len(regions) == 5+11+18+24+30+36+1


    @pytest.fixture
    def sim_params(self):
        return SimulationParams(runid="run0406_ID000126", onset=250, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1")

    @pytest.fixture
    def online_photometry(self, sim_params):
        return OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)

    @pytest.fixture
    def output_dir(self):
        output_dir_path = Path(__file__).absolute().parent.joinpath("test_photometry3_output")
        output_dir_path.mkdir(exist_ok=True, parents=True)
        return output_dir_path

    def test_online_photometry_preconfigured_no_target(self, online_photometry, output_dir):
        add_target_region = False
        template = False
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.5, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=False)

    def test_online_photometry_preconfigured_with_target(self, online_photometry, output_dir):
        add_target_region = True
        template = "run0406_ID000126"
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.5, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=False)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir, template)

    def test_online_photometry_preconfigured_with_target_remove_overlapping(self, online_photometry, output_dir):
        add_target_region = True
        remove_overlapping_regions_with_target = True        
        template = "run0406_ID000126"
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.5, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir, template)





    """
    def test_integrate(self):
        
        data_dir = "/data01/homes/baroncelli/phd/tests/test_data/fits-prod5/bkg_only"
        config_file = Path(data_dir).joinpath("config.yaml")
        photon_list = Path(data_dir).joinpath("runid_notemplate_trial_0001024120_simtype_bkg_onset_0_delay_0_offset_0.5.fits")
        
        pht = OnlinePhotometry(config_file)

        region_radius = 0.2
        integration_time = 5
        number_of_energy_bins = 3
        max_points = 5
        reflection = True
        normalize = True
        threads = 10
        max_offset = 2

        s = time()
        regions_dict = pht.create_photometry_configuration(region_radius, number_of_energy_bins, configuration="reflected", max_offset=max_offset)
        print(f"Time to create the regions configuration: {time() - s} sec")

        s = time()
        data, data_err = pht.integrate(photon_list, regions_dict, region_radius, integration_time, number_of_energy_bins, max_points, normalize=normalize, threads=threads)
        print(f"Time to integrate: {time() - s} sec")

        assert data.shape == (56, 5, 3)
        print(data[0])

        threads = 1
        s = time()
        data, data_err = pht.integrate(photon_list, regions_dict, region_radius, integration_time, number_of_energy_bins, max_points, normalize=normalize, threads=threads)
        print(f"Time to integrate: {time() - s} sec")


        normalize = False
        threads = 10
        s = time()
        data, data_err = pht.integrate(photon_list, regions_dict, region_radius, integration_time, number_of_energy_bins, max_points, normalize=normalize, threads=threads)
        print(data[0])
        print(f"Time to integrate: {time() - s} sec")
    """