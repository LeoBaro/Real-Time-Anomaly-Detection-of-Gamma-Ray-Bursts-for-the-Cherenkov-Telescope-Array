from rtapipe.analysis.models.anomaly_detector_m1 import AnomalyDetector_m1
from rtapipe.analysis.models.anomaly_detector_m2 import AnomalyDetector_m2
from rtapipe.analysis.models.anomaly_detector_m3 import AnomalyDetector_m3
from rtapipe.analysis.models.anomaly_detector_m4 import AnomalyDetector_m4

class AnomalyDetectorBuilder:

    @staticmethod
    def getAnomalyDetector(name, timesteps, nfeatures, outDir, loadModel = False, modelDir = None):

        if name == "m1":
            if loadModel:
                return AnomalyDetector_m1.loadModel(modelDir)
            return AnomalyDetector_m1(timesteps, nfeatures, outDir, loadModel)

        elif name == "m2":
            if loadModel:
                return AnomalyDetector_m2.loadModel(modelDir)
            return AnomalyDetector_m2(timesteps, nfeatures, outDir, loadModel)

        elif name == "m3":
            if loadModel:
                return AnomalyDetector_m3.loadModel(modelDir)
            return AnomalyDetector_m3(timesteps, nfeatures, outDir, loadModel)

        elif name == "m4":
            if loadModel:
                return AnomalyDetector_m4.loadModel(modelDir)
            return AnomalyDetector_m4(timesteps, nfeatures, outDir, loadModel)

        else:
            raise ValueError("Model name not supported!")


    @staticmethod
    def getTrainingParams(trainingType):
        if trainingType == "light":
            return {
                "sample_size" : 16,
                "batch_size" : 4
            }
        elif trainingType == "medium":
            return {
                "sample_size" : 256,
                "batch_size" : 16
            }
        elif trainingType == "heavy":
            return {
                "sample_size" : 4096,
                "batch_size" : 64
            }
        else:
            raise ValueError()
