import pytest
from pathlib import Path

from rtapipe.lib.dataset.dataset import APDataset
from rtapipe.lib.plotting.plotting import plot_sequences

class TestAPDataset:

    def test_multiple_phlist(self):
        
        # make me a fixture
        conf_file = Path(Path(__file__).resolve().parent).joinpath("conf/agilehost3-prod5-test.yml")

        ds = APDataset.get_dataset(conf_file, 101, out_dir="./test_multiple_phlist-out", scaler_type="mm")

        ds.loadData()

        assert ds.filesLoaded == 2 
        assert ds.data.shape == (20, 4)

        train_x, train_y, val_x, val_y = ds.train_val_split(split=50)

        assert train_x.shape == (1, 10, 4)
        assert val_x.shape == (1, 10, 4)

        sample = ds.get_random_train_sample(True)
        assert sample.shape == (10, 4)


    def test_single_phlist(self):

        # make me a fixture
        conf_file = Path(Path(__file__).resolve().parent).joinpath("conf/agilehost3-prod5-test.yml")

        ds = APDataset.get_dataset(conf_file, 1201, out_dir="./test_single_phlist-out", scaler_type="mm")

        ds.loadData()

        assert ds.filesLoaded == 1
        assert ds.data.shape == (3600, 3)

        train_x, train_y, val_x, val_y = ds.train_val_split(tsl=10, split=50, stride=10)

        assert train_x.shape == (180, 10, 3)
        assert val_x.shape == (180, 10, 3)


    def test_single_phlist_test_set(self):
        
        conf_file = Path(Path(__file__).resolve().parent).joinpath("conf/agilehost3-prod5-test.yml")

        ds = APDataset.get_dataset(conf_file, 1201, out_dir="./test_single_phlist_test_set-out", scaler_type="mm")

        ds.loadData()

        train_x, train_y, val_x, val_y = ds.train_val_split(tsl=10, split=50, stride=10)

        ds.load_test_data("test_set_a") # single phtoon list
        test_x, test_y = ds.get_test_set(stride=1)

        assert test_x.shape == (91, 10, 3)
        assert test_y.shape == (91,)


        plot_sequences(test_x[:5], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_0-5.png")
        plot_sequences(test_x[46:51], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_46-51.png")
        plot_sequences(test_x[50:55], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_50-55.png")
        plot_sequences(test_x[60:65], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_60-65.png")

    def test_single_phlist_test_set_multiple_pht_lists(self):
        
        conf_file = Path(Path(__file__).resolve().parent).joinpath("conf/agilehost3-prod5-test.yml")

        ds = APDataset.get_dataset(conf_file, 1201, out_dir="./test_single_phlist_test_set_multiple_pht_lists-out", scaler_type="mm")

        ds.loadData()

        train_x, train_y, val_x, val_y = ds.train_val_split(tsl=10, split=50, stride=10)

        ds.load_test_data("test_set_b") # single phtoon list
        test_x, test_y = ds.get_test_set(stride=1)
        
        assert test_x.shape == (182, 10, 3)
        assert test_y.shape == (182,)

        plot_sequences(test_x[:5], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set_multiple_pht_lists-out", figName="samples_0-5.png")
        plot_sequences(test_x[46:51], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set_multiple_pht_lists-out", figName="samples_46-51.png")
        plot_sequences(test_x[50:55], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set_multiple_pht_lists-out", figName="samples_50-55.png")
        plot_sequences(test_x[60:65], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set_multiple_pht_lists-out", figName="samples_60-65.png")        

        plot_sequences(test_x[91:96], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set_multiple_pht_lists-out", figName="samples_91-96.png")
        plot_sequences(test_x[137:142], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set_multiple_pht_lists-out", figName="samples_137-142.png")
