import yaml
from pathlib import Path
from tensorflow import keras

from rtapipe.lib.utils.misc import dotdict
from rtapipe.lib.evaluation.pval import get_pval_table
from rtapipe.lib.models.anomaly_detector_lstm import *
from rtapipe.lib.models.anomaly_detector_rnn import *
from rtapipe.lib.models.anomaly_detector_cnn import *

class AnomalyDetectorBuilder:

    @staticmethod
    def getModelsName(model_type=None):
        models = [class_name for class_name in globals() if "AnomalyDetector_" in class_name ]
        if model_type is not None:
            models = [model for model in models if model_type in model]
        return models

    @staticmethod
    def load_model(model_id):
        
        with open(Path(__file__).parent.joinpath("trained_models.yaml"), "r") as f:
            try:
                configs = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        model_config = dotdict([c for c in configs["models"] if c["id"] == model_id].pop())        
        model_config["ad"] = AnomalyDetectorBuilder.getAnomalyDetector(name=model_config["name"], timesteps=model_config["timesteps"], nfeatures=model_config["nfeatures"], load_model="True", training_epoch_dir=model_config["path"], training=False)
        model_config["pvalue_table"] = get_pval_table(model_config["pval_path"]) 
        return model_config



    @staticmethod
    def getAnomalyDetector(name, timesteps, nfeatures, load_model=False, training_epoch_dir=None, training=True):
        
        if timesteps is None or nfeatures is None:
            raise Exception("timesteps and nfeatures are required")
            
        klass = globals()[name]

        if not training and not Path(training_epoch_dir).exists():
            raise Exception(f"Training epoch dir {training_epoch_dir} does not exist")

        if not training:
            model_dir = Path(training_epoch_dir).joinpath("trained_model")
            if not model_dir.exists():
                raise Exception(f"Model dir {model_dir} does not exist")

            threshold_file = Path(training_epoch_dir).joinpath("threshold.txt")
            if not threshold_file.exists():
                raise Exception(f"Threshold file {threshold_file} does not exist")

            with open(threshold_file, "r") as thf:
                threshold = float(thf.read().rstrip().strip())

        if load_model:
            ad = klass(timesteps, nfeatures, loadModel=True, threshold=threshold)
            ad.model = keras.models.load_model(model_dir, compile=training)
            return ad

        return klass(timesteps, nfeatures, False, None)


    @staticmethod
    def getTrainingParams(trainingType):
        if trainingType == "light":
            return {
                "sample_size" : 1024,
                "batch_size" : 4
            }
        elif trainingType == "medium":
            return {
                "sample_size" : 1024,
                "batch_size" : 8
            }
        elif trainingType == "heavy":
            return {
                "sample_size" : 1024,
                "batch_size" : 16
            }
        else:
            raise ValueError()

