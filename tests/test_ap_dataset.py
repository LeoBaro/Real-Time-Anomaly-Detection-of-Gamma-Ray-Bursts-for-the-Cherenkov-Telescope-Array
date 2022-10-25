import pytest
from pathlib import Path

from rtapipe.lib.dataset.dataset import APDataset

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

        ds.plotRandomSample(howMany=10)
        
        assert Path("./test_multiple_phlist-out/random_sample_scaled_True.png").exists()

        ds.plotRandomSample(howMany=10, scaled=False)

        assert Path("./test_multiple_phlist-out/random_sample_scaled_False.png").exists()

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

        ds.plotRandomSample(howMany=10)

        assert Path("./test_single_phlist-out/random_sample_scaled_True.png").exists()

        ds.plotRandomSample(howMany=10, scaled=False)

        assert Path("./test_single_phlist-out/random_sample_scaled_False.png").exists()
