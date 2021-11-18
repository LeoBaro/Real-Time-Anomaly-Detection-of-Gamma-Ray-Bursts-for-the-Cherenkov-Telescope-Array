import pytest
from pathlib import Path

from rtapipe.analysis.dataset.dataset import APDataset

class TestDataset:

    def test_loadDataset(self):

        ds = APDataset.get_dataset(1, "../analysis/dataset/config/agilehost3.yml",scaler="mm", outDir="./test-out-tmp")

    def test_loadDatasetBatch(self):

        
