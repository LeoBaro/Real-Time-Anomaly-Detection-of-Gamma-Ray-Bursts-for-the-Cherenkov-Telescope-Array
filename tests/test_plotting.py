import numpy as np
import tensorflow as tf
from pathlib import Path
from rtapipe.lib.plotting.plotting import plot_predictions

class TestPlotting:

    def test_plot_predictions_v2(self):
        
        samples = np.random.uniform(0, 1, size=(10, 5, 3))
        samplesLabels = np.random.randint(0,2, size=(10))
        c_threshold = 0.54321
        recostructions = np.random.uniform(0, 0.5, size=(10, 5, 3)) 
        mse_per_sample = np.random.uniform(0, 1, size=(10))
        mse_per_sample_features = np.random.uniform(0, 1, size=(10, 3))
        max_plots=10
        showFig=False
        saveFig=True 
        outputDir=Path(__file__).parent.joinpath("test_plot_predictions_output")
        
        plot_predictions(samples, samplesLabels, c_threshold, recostructions, mse_per_sample, mse_per_sample_features, max_plots, showFig, saveFig, outputDir)

