from rtapipe.analysis.models.anomaly_detector_base import AnomalyDetectorBase

class AnomalyDetector_m3(AnomalyDetectorBase):

    @staticmethod
    def loadModel(modelDir):
        try:
            print(f"Loading model from {modelDir}")
            ad = AnomalyDetector_m3(0, 0, Path(modelDir).parent, True)
            ad.model = load_model(modelDir)
            return ad
        except Exception:
            print(f"Unable to load model from {modelDir}.")

    def __init__(self, timesteps, nfeatures, outDir, loadModel = False):
        super().__init__(timesteps, nfeatures, outDir, loadModel)
        
