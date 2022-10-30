import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import numpy as np
from rtapipe.lib.models.anomaly_detector_base import AnomalyDetectorBase

class TestAnomalyDetectorBase:

    def test_detection_delay(self):

        num_samples = 96
        number_of_files = 10
        tsl = 5
        y = []
        y_pred = []
        for nf in range(number_of_files):
            y += [False for i in range(num_samples//2)] + [True for i in range(num_samples//2)]
            y_pred += [False for i in range((num_samples//2)+2)] + [True for i in range((num_samples//2)-2)]
        y = np.array(y)
        y_pred = np.array(y_pred)
        assert y.shape == (960,)
        assert y_pred.shape == (960,)

        adb = AnomalyDetectorBase(None,None,None)

        adb.detection_delay(y, y_pred, number_of_files, tsl)
