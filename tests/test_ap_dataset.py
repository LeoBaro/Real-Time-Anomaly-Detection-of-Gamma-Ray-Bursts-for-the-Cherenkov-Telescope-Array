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
        assert ds.data.shape == (1800, 3)

        train_x, train_y, val_x, val_y = ds.train_val_split(tsl=10, split=50)

        assert train_x.shape == (90, 10, 3)
        assert val_x.shape == (90, 10, 3)


    def test_single_phlist_test_set(self):
        
        conf_file = Path(Path(__file__).resolve().parent).joinpath("conf/agilehost3-prod5-test.yml")

        ds = APDataset.get_dataset(conf_file, 1201, out_dir="./test_single_phlist_test_set-out", scaler_type="mm")

        ds.loadData()

        train_x, train_y, val_x, val_y = ds.train_val_split(tsl=10, split=50)
        test_x, test_y = ds.get_test_set(tsl=10, stride=1)

        plot_sequences(test_x[:5], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_0-5.png")
        plot_sequences(test_x[46:51], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_46-51.png")
        plot_sequences(test_x[50:55], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_50-55.png")
        plot_sequences(test_x[60:65], scaled=True, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./test_single_phlist_test_set-out", figName="samples_60-65.png")