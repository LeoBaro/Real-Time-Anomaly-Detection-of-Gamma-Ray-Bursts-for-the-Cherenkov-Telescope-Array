import os
import pytest
from pathlib import Path

from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils
from rtapipe.lib.datasource.Photometry3 import RegionsConfig, SimulationParams, Region, OnlinePhotometry


@pytest.fixture
def irf():
    os.environ["CTOOLS"] = "/data01/homes/baroncelli/.conda/envs/bphd"
    irf = Path(os.environ['CTOOLS']).joinpath("share","caldb","data","cta","prod5-v0.1","bcf","North_z40_5h_LST")
    return irf.joinpath(os.listdir(irf).pop())

@pytest.fixture
def sim_params(request):
    return SimulationParams(runid=request.param, simtype="grb", onset=250, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1")


@pytest.fixture
def output_dir():
    output_dir_path = Path(__file__).absolute().parent.joinpath("test_photometry3_output")
    output_dir_path.mkdir(exist_ok=True, parents=True)
    return output_dir_path


class TestPhotometry3:


    def test_region_config_rings_regions_without_target(self, irf):

        rc = RegionsConfig(0.2, 2)

        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0})
        
        assert len(rc.rings.keys()) == 5 
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0]
        assert len(rc.rings[0.4]) == 5
        assert len(rc.rings[0.8]) == 11
        assert len(rc.rings[1.2]) == 17
        assert len(rc.rings[1.6]) == 23
        assert len(rc.rings[2.0]) == 29

        regions = rc.get_flatten_configuration(regions_type="src")
        assert len(regions) == 0
        regions = rc.get_flatten_configuration(regions_type="bkg")
        assert len(regions) == 5+11+17+23+29       
        regions = rc.get_flatten_configuration(regions_type="all")
        assert len(regions) == 5+11+17+23+29

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
        assert len(rc.rings[0.4]) == 5
        assert len(rc.rings[0.8]) == 11
        assert len(rc.rings[1.2]) == 17
        assert len(rc.rings[1.6]) == 23
        assert len(rc.rings[2.0]) == 29

        regions = rc.get_flatten_configuration(regions_type="src")
        assert len(regions) == 1

        regions = rc.get_flatten_configuration(regions_type="bkg")
        assert len(regions) == 5+11+17+23+29       

        regions = rc.get_flatten_configuration(regions_type="all")
        assert len(regions) == 5+11+17+23+29+1  

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
            assert list(rc.target_region.effective_area.keys())[i] == energy_windows[i]



    def test_region_config_rings_regions_with_target_no_overlaps(self, irf):

        rc = RegionsConfig(0.2, 2)
        remove_overlapping_regions_with_target = True
        rc.compute_rings_regions(pointing={"ra":20.0, "dec": -50.0}, add_target_region={"ra":19.5, "dec": -50.0}, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target)
        
        assert len(rc.rings.keys()) == 5
        assert list(rc.rings.keys()) == [0.4, 0.8, 1.2, 1.6, 2.0]
        assert len(rc.rings[0.4]) == 4
        assert len(rc.rings[0.8]) == 10
        assert len(rc.rings[1.2]) == 17
        assert len(rc.rings[1.6]) == 23
        assert len(rc.rings[2.0]) == 29

        regions = rc.get_flatten_configuration(regions_type="src")
        assert len(regions) == 1
        regions = rc.get_flatten_configuration(regions_type="bkg")
        assert len(regions) == 4+10+17+23+29
        regions = rc.get_flatten_configuration(regions_type="all")
        assert len(regions) == 4+10+17+23+29+1

        energy_windows = PhotometryUtils.getLogWindows(0.04, 1, 3)

        rc.compute_effective_area(irf, {"ra":20.0, "dec": -50.0}, energy_windows)        
        for regions in rc.rings.values():
            for region in regions:
                assert len(region.effective_area.keys()) == 3
                for i in range(3):
                    assert list(region.effective_area.keys())[i] == energy_windows[i]
                    assert region.effective_area[energy_windows[i]] > 0

        for i in range(3):
            assert list(rc.target_region.effective_area.keys())[i] == energy_windows[i]


    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_online_photometry_preconfigured_no_target(self, sim_params, output_dir):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        add_target_region = False
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        compute_effective_area_for_normalization=False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=False, compute_effective_area_for_normalization=compute_effective_area_for_normalization)

    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_online_photometry_preconfigured_with_target(self, sim_params, output_dir):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        add_target_region = True
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        compute_effective_area_for_normalization=False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=False, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir)

    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_online_photometry_preconfigured_with_target_remove_overlapping(self, sim_params, output_dir):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)        
        add_target_region = True
        remove_overlapping_regions_with_target = True        
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        compute_effective_area_for_normalization=False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir)
    
    @pytest.mark.parametrize("sim_params", ['run0001_ID000048'], indirect=True)
    def test_online_photometry_with_different_template(self, sim_params, output_dir):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        add_target_region = True
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_test/sim_output/run0001_ID000048/runid_run0001_ID000048_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        compute_effective_area_for_normalization=True
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=True, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        online_photometry.generate_skymap_with_regions(test_fits, output_dir)


    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_bkg_regions_with_preconfigured_regions_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        compute_effective_area_for_normalization = True
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        normalize = True
        with_metadata = True
        integrate_from_regions = "bkg"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, integrate_from_regions=integrate_from_regions)

        assert data.shape == (83, 5, 3)
        assert data_err.shape == (83, 5, 3)
        assert metadata["number_of_regions"] == 83
        print("integration elapsed time: ", metadata["elapsed_time"])

    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_bkg_regions_with_no_preconfigured_regions_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        normalize = True
        with_metadata = True
        integrate_from_regions = "bkg"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, integrate_from_regions=integrate_from_regions)

        assert data.shape == (83, 5, 3)
        assert data_err.shape == (83, 5, 3)
        assert metadata["number_of_regions"] == 83
        print("integration elapsed time: ", metadata["elapsed_time"])

    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_bkg_regions_with_preconfigured_regions_not_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        compute_effective_area_for_normalization = False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        normalize = False
        with_metadata = True
        integrate_from_regions = "bkg"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, integrate_from_regions=integrate_from_regions)

        assert data.shape == (83, 5, 3)
        assert data_err.shape == (83, 5, 3)
        assert metadata["number_of_regions"] == 83
        print("integration elapsed time: ", metadata["elapsed_time"])


    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_bkg_regions_with_no_preconfigured_regions_not_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        normalize = False
        with_metadata = True
        integrate_from_regions = "bkg"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, integrate_from_regions=integrate_from_regions)

        assert data.shape == (83, 5, 3)
        assert data_err.shape == (83, 5, 3)
        assert metadata["number_of_regions"] == 83
        print("integration elapsed time: ", metadata["elapsed_time"])







    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_src_regions_with_preconfigured_regions_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        compute_effective_area_for_normalization = True
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        normalize = True
        with_metadata = True
        integrate_from_regions = "src"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, integrate_from_regions=integrate_from_regions)

        assert data.shape == (1, 5, 3)
        assert data_err.shape == (1, 5, 3)
        assert metadata["number_of_regions"] == 1
        print("integration elapsed time: ", metadata["elapsed_time"])

    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_src_regions_with_no_preconfigured_regions_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        normalize = True
        with_metadata = True
        integrate_from_regions = "src"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, integrate_from_regions=integrate_from_regions)

        assert data.shape == (1, 5, 3)
        assert data_err.shape == (1, 5, 3)
        assert metadata["number_of_regions"] == 1
        print("integration elapsed time: ", metadata["elapsed_time"])

    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_src_regions_with_preconfigured_regions_not_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        compute_effective_area_for_normalization = False
        online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization)
        normalize = False
        with_metadata = True
        integrate_from_regions = "src"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, integrate_from_regions=integrate_from_regions)

        assert data.shape == (1, 5, 3)
        assert data_err.shape == (1, 5, 3)
        assert metadata["number_of_regions"] == 1
        print("integration elapsed time: ", metadata["elapsed_time"])


    @pytest.mark.parametrize("sim_params", ['run0406_ID000126'], indirect=True)
    def test_integrate_src_regions_with_no_preconfigured_regions_not_normalized(self, sim_params):
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        test_fits = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000002_simtype_grb_onset_250_delay_0_offset_0.5.fits"
        add_target_region = True
        remove_overlapping_regions_with_target = True  
        normalize = False
        with_metadata = True
        integrate_from_regions = "src"
        data, data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, integrate_from_regions=integrate_from_regions)

        assert data.shape == (1, 5, 3)
        assert data_err.shape == (1, 5, 3)
        assert metadata["number_of_regions"] == 1
        print("integration elapsed time: ", metadata["elapsed_time"])




