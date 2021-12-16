import argparse
import matplotlib as pyplot

from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-di", "--dataset_id", type=int, required=True, help="", choices=[1,2,3,4])
    parser.add_argument('-sa', '--save-after', type=int, required=False, default=1, help="Store trained model after sa training epochs")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="./dataset/config/agilehost3.yml")
    args = parser.parse_args()

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Output dir
    outDirRoot = Path(__file__)
                    .parent
                    .resolve()
                    .joinpath("training_output",f"lstm_models_{strftime('%Y%m%d-%H%M%S')}")
