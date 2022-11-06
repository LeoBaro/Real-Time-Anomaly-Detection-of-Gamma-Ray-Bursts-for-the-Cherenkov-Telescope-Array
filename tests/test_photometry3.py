import os
import pytest
from pathlib import Path

from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils
from rtapipe.lib.datasource.Photometry3 import RegionsConfig, SimulationParams, Region, OnlinePhotometry

class TestPhotometry3:

    @pytest.fixture
    def irf(self):
        os.environ["CTOOLS"] = "/data01/homes/baroncelli/.conda/envs/bphd"
        irf = Path(os.environ['CTOOLS']).joinpath("share","caldb","data","cta","prod5-v0.1","bcf","North_z40_5h_LST")
        return irf.joinpath(os.listdir(irf).pop())

    def test_region_config_rings_regions_without_target(self, irf):

        rc = RegionsConfig(0.2, 2)

        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0})
        
        assert len(rc.rings.keys()) == 5 
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0]
        assert len(rc.rings[0.4]) == 6
        assert len(rc.rings[0.8]) == 12
        assert len(rc.rings[1.2]) == 18
        assert len(rc.rings[1.6]) == 24
        assert len(rc.rings[2.0]) == 30
        
        regions = rc.get_flatten_configuration()
        assert len(regions) == 6+12+18+24+30

        energy_windows = PhotometryUtils.getLogWindows(0.04, 1, 3)

        rc.compute_effective_area(irf, {"ra":20.0, "dec": -50.0}, energy_windows)        
        for regions in rc.rings.values():
            for region in regions:
                assert len(region.effective_area.keys()) == 3
                for i in range(3):
                    assert list(region.effective_area.keys())[i] == energy_windows[i]
                    assert region.effective_area[energy_windows[i]] > 0

        assert rc.target_region is None

    def test_region_config_rings_regions_with_target(self, irf):

        rc = RegionsConfig(0.2, 2)

        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0}, add_target_region={"ra":19.5, "dec": -50.0})
        
        assert len(rc.rings.keys()) == 5
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0]
        assert len(rc.rings[0.4]) == 6
        assert len(rc.rings[0.8]) == 12
        assert len(rc.rings[1.2]) == 18
        assert len(rc.rings[1.6]) == 24
        assert len(rc.rings[2.0]) == 30

        regions = rc.get_flatten_configuration()
        assert len(regions) == 6+12+18+24+30

        assert rc.target_region is not None

        energy_windows = PhotometryUtils.getLogWindows(0.04, 1, 3)

        rc.compute_effective_area(irf, {"ra":20.0, "dec": -50.0}, energy_windows)        
        for regions in rc.rings.values():
            for region in regions:
                assert len(region.effective_area.keys()) == 3
                for i in range(3):
                    assert list(region.effective_area.keys())[i] == energy_windows[i]
                    assert region.effective_area[energy_windows[i]] > 0

        for i in range(3):
            assert list(rc.target_region_effective_area.keys())[i] == energy_windows[i]



    def test_region_config_rings_regions_with_target_no_overlaps(self, irf):

        rc = RegionsConfig(0.2, 2)
        remove_overlapping_regions_with_target = True
        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0}, add_target_region={"ra":19.5, "dec": -50.0}, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)
        
        assert len(rc.rings.keys()) == 5
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0]
        assert len(rc.rings[0.4]) == 5
        assert len(rc.rings[0.8]) == 11
        assert len(rc.rings[1.2]) == 18
        assert len(rc.rings[1.6]) == 24
        assert len(rc.rings[2.0]) == 30

        regions = rc.get_flatten_configuration()
        assert len(regions) == 5+11+18+24+30

        energy_windows = PhotometryUtils.getLogWindows(0.04, 1, 3)

        rc.compute_effective_area(irf, {"ra":20.0, "dec": -50.0}, energy_windows)        
        for regions in rc.rings.values():
            for region in regions:
                assert len(region.effective_area.keys()) == 3
                for i in range(3):
                    assert list(region.effective_area.keys())[i] == energy_windows[i]
                    assert region.effective_area[energy_windows[i]] > 0

        for i in range(3):
            assert list(rc.target_region_effective_area.keys())[i] == energy_windows[i]



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
        compute_effective_area_for_normalization=False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=False, compute_effective_area_for_normalization=compute_effective_area_for_normalization)

    def test_online_photometry_preconfigured_with_target(self, online_photometry, output_dir):
        add_target_region = True
        template = "run0406_ID000126"
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        compute_effective_area_for_normalization=False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=False, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir, template)

    def test_online_photometry_preconfigured_with_target_remove_overlapping(self, online_photometry, output_dir):
        add_target_region = True
        remove_overlapping_regions_with_target = True        
        template = "run0406_ID000126"
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        compute_effective_area_for_normalization=False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir, template)
    
    
    def test_integrate_with_preconfigured_regions_normalized(self, online_photometry):
        
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        template = "run0406_ID000126"
        compute_effective_area_for_normalization = True
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        normalize = True
        with_metadata = True
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata)

        assert data.shape == (86, 5, 3)
        assert data_err.shape == (86, 5, 3)
        assert metadata["number_of_regions"] == 86
        print("integration elapsed time: ", metadata["elapsed_time"])


    def test_integrate_with_no_preconfigured_regions_normalized(self, online_photometry):
        
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        template = "run0406_ID000126"
        normalize = True
        with_metadata = True
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)

        assert data.shape == (86, 5, 3)
        assert data_err.shape == (86, 5, 3)
        assert metadata["number_of_regions"] == 86
        print("integration elapsed time: ", metadata["elapsed_time"])


    def test_integrate_with_preconfigured_regions_not_normalized(self, online_photometry):
        
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        template = "run0406_ID000126"
        compute_effective_area_for_normalization = False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        normalize = False
        with_metadata = True
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata)

        assert data.shape == (86, 5, 3)
        assert data_err.shape == (86, 5, 3)
        assert metadata["number_of_regions"] == 86
        print("integration elapsed time: ", metadata["elapsed_time"])



    def test_integrate_with_no_preconfigured_regions_not_normalized(self, online_photometry):
        
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        template = "run0406_ID000126"
        normalize = False
        with_metadata = True
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, template=template, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)

        assert data.shape == (86, 5, 3)
        assert data_err.shape == (86, 5, 3)
        assert metadata["number_of_regions"] == 86
        print("integration elapsed time: ", metadata["elapsed_time"])

