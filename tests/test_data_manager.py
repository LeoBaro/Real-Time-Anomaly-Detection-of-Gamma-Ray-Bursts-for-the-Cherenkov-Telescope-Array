import pytest
from pathlib import Path
from rtapipe.lib.dataset.data_manager import DataManager
from rtapipe.lib.datasource.Photometry3 import SimulationParams

class TestDataManager:

    @pytest.fixture
    def output_dir(self):
        output_dir_path = Path(__file__).absolute().parent.joinpath("test_data_manager_output")
        output_dir_path.mkdir(exist_ok=True, parents=True)
        return output_dir_path
        
    @pytest.fixture
    def data_manager(self, output_dir):
        return DataManager(output_dir)

    def test_load_fits_data(self):
        dataset_folder_1 = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data"
        dataset_folder_2 = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_c/fits_data"
        fits_files = DataManager.load_fits_data(dataset_folder_1)
        assert len(fits_files) == 10
        fits_files = DataManager.load_fits_data(dataset_folder_2, limit=7)
        assert len(fits_files) == 7

    def test_extract_runid_from_name(self):
        name = "'/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data/runid_run0406_ID000126_trial_0000000007_simtype_grb_onset_250_delay_0_offset_0.5.fits'"
        runid = DataManager.extract_runid_from_name(name)
        assert runid == "run0406_ID000126"

    def test_transform_to_timeseries_single_template_with_bkg(self, data_manager, output_dir):
        dataset_folder = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/train/North_z40_5h_LST/itime_5_b/fits_data"
        multiple_templates = False
        template = "notemplate"
        add_target_region = False
        fits_files = DataManager.load_fits_data(dataset_folder, limit=1)
        sim_params = SimulationParams(runid=template, onset=250, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1", simtype="bkg")
        data_manager.transform_to_timeseries(fits_files, sim_params, add_target_region, integration_time=5, number_of_energy_bins=3, tsl=100, normalize=True, threads=20, multiple_templates=multiple_templates)
        assert len(list(data_manager.data.keys())) == 1
        assert template in data_manager.data
        assert data_manager.data[template].shape == (85, 100, 3) # (1 trial * 85 regions, 100 points, 3 features)
        DataManager.plot_timeseries(template, data_manager.data[template], 10, sim_params, output_dir)

    def test_transform_to_timeseries_single_template_with_src(self, data_manager, output_dir):
        dataset_folder = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_b/fits_data"
        multiple_templates = False
        template = "run0406_ID000126"
        add_target_region = True
        fits_files = DataManager.load_fits_data(dataset_folder)
        sim_params = SimulationParams(runid=template, onset=250, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1", simtype="grb")
        data_manager.transform_to_timeseries(fits_files, sim_params, add_target_region, integration_time=5, number_of_energy_bins=3, tsl=100, normalize=True, threads=20, multiple_templates=multiple_templates)
        assert len(list(data_manager.data.keys())) == 1
        assert template in data_manager.data
        assert data_manager.data[template].shape == (10, 100, 3) # (10 trials * 1 regions, 100 points, 3 features)
        DataManager.plot_timeseries(template, data_manager.data[template], 10, sim_params, output_dir)

    def test_transform_to_timeseries_multiple_templates_with_src(self, data_manager, output_dir):
        """
        BUG DEI TEMPLATES!!
        """
        dataset_folder = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_c/fits_data"
        multiple_templates = True
        add_target_region = True
        fits_files = DataManager.load_fits_data(dataset_folder, limit=50)
        sim_params = SimulationParams(runid=None, onset=250, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1", simtype="grb")
        data_manager.transform_to_timeseries(fits_files, sim_params, add_target_region, integration_time=5, number_of_energy_bins=3, tsl=100, normalize=True, threads=20, multiple_templates=multiple_templates)
        assert len(list(data_manager.data.keys())) == 50
        for template_name, data in data_manager.data.items():
            print("template_name", template_name)
            assert data.shape == (1, 100, 3) # (1 trial * 1 region, 100 points, 3 features)
            sim_params.runid = template_name
            DataManager.plot_timeseries(template_name, data, 1, sim_params, output_dir)






    def test_get_train_and_test_set(self, data_manager, output_dir):
        """
        Real example:
        - fits file of the training set are loaded
        - 5 trial of the same template (bkg only) with tobs 500 seconds
        - the trials are integrated in 5 seconds => 100 points
        - the trials are integrated using data from 90 regions of interest => 90 samples * 5 = 450 samples
        - from each sample are then extracted subsequences of 5 points with stride 5 => ( (100-5)/5 + 1 ) * 450 = 9000 samples   
        - the samples are then splitted in train and validation sets with ratio 0.5/0.5 
        - a MinMaxScaler is applied to the data     
        then:
        - new fits files are loaded (test set)
        - 5 trials of different templates (bkg+grb) with tobs 500 seconds and onset=250
        - the trials are integrated in 5 seconds => 100 points
        - the trials are integrated using data from 1 region of interest => 1 samples * 5 = 5 samples
        - from the first template are then extracted subsequences of 5 points with stride 1 => ( (100-5)/1 + 1 ) * 90 = 8640 samples   
        - a the MinMaxScaler fitted on the training set, is applied to the subsequences
        """
        dataset_folder = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/train/North_z40_5h_LST/itime_5_b/fits_data"
        fits_files = DataManager.load_fits_data(dataset_folder, limit=5)
        sim_params = SimulationParams(runid=None, onset=0, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1", simtype="bkg")
        multiple_templates = False
        add_target_region = False
        data_manager.transform_to_timeseries(fits_files, sim_params, add_target_region, integration_time=5, number_of_energy_bins=3, tsl=100, normalize=True, threads=30, multiple_templates=multiple_templates)
        #data_manager.load_saved_data("notemplate", 5, 100)
        assert data_manager.data["notemplate"].shape == (425, 100, 3)
        
        train_x, train_y , val_x, val_y = data_manager.get_train_set("notemplate", sub_window_size=5, stride=5, validation_split=50)
        assert train_x.shape == (4250, 5, 3)
        assert train_y.shape == (4250,)
        assert val_x.shape == (4250, 5, 3)
        assert val_y.shape == (4250,)

        assert train_x[0].min() >= 0
        assert train_x[0].max() <= 1.0
        assert val_x[0].min() >= 0
        assert val_x[0].max() <= 1.0

        # DataManager.plot_timeseries("notemplate", train_x, 5, sim_params, output_dir, max_flux=0.6)

 

        test_dataset_folder = "/data01/homes/baroncelli/phd/rtapipe/scripts/ml/dataset_generation/test/itime_5_c/fits_data"
        fits_files = DataManager.load_fits_data(test_dataset_folder, limit=5) # run0001_ID000048, run0001_ID000165, run0002_ID000044, run0002_ID000185
        multiple_templates = True
        add_target_region = True
        sim_params = SimulationParams(runid=None, onset=250, emin=0.04, emax=1, tmin=0, tobs=500, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1", simtype="grb")
        data_manager.transform_to_timeseries(fits_files, sim_params, add_target_region, integration_time=5, number_of_energy_bins=3, tsl=100, normalize=True, threads=20, multiple_templates=multiple_templates)
        #data_manager.load_saved_data("run0001_ID000001", 5, 100)
        
        assert data_manager.data["run0001_ID000001"].shape == (1, 100, 3)

        test_x, test_y = data_manager.get_test_set(template="run0001_ID000001", onset=250, integration_time=5, sub_window_size=5, stride=5)

        assert test_x.shape == (20, 5, 3)
        assert test_y.shape == (20,)

        assert test_x[0].min() >= 0
        assert test_x[0].max() <= 1.0

        DataManager.plot_timeseries("run0001_ID000001", test_x, 20, sim_params, output_dir, max_flux=1.5)
