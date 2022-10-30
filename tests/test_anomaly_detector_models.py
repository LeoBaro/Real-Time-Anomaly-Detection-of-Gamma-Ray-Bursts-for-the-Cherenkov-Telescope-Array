import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import numpy as np
from rtapipe.lib.models.anomaly_detector_builder import AnomalyDetectorBuilder
from rtapipe.lib.evaluation.custom_mse import CustomMSE

class TestAnomalyDetectorModels:

    def test_tsl_5_nfeatures_3(self):
        
        _timesteps = [5, 10]
        _nfeatures = [3]

        models = AnomalyDetectorBuilder.getModelsName()
        #models = AnomalyDetectorBuilder.getModelsName("cnn")
        #models = AnomalyDetectorBuilder.getModelsName("rnn")
        #models = AnomalyDetectorBuilder.getModelsName("lstm")

        for timesteps in _timesteps:

            for nfeatures in _nfeatures:

                samples = np.random.uniform(0, 1, size=(10, timesteps, nfeatures))

                for model_name in models:

                    model = AnomalyDetectorBuilder.getAnomalyDetector(model_name, timesteps, nfeatures, load_model=False, training_epoch_dir=None, training=True)
                    assert model.timesteps == timesteps
                    assert model.nfeatures == nfeatures
                    assert model.model is not None

                    model.model.compile(optimizer='adam', loss=CustomMSE(nfeatures, output_dir="/tmp"))

                    
                    model.model.fit(samples, samples, verbose=0, epochs=1, batch_size=10, validation_data=None, callbacks=None)

 