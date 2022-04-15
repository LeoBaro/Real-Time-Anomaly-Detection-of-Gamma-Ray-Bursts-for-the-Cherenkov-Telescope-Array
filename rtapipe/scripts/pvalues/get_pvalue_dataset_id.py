import os
import yaml
import argparse
from pathlib import Path

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--line", type=str, required=True, help="A line such as id=400")
    args = parser.parse_args()

    dataset_config_path = Path(os.environ["DATASET_CONFIG_FILE"])

    # Read YAML file
    with open(dataset_config_path, 'r') as stream:
        dataset_config = yaml.safe_load(stream)


    datasetid = args.line.split("=")[1]
    pval_datasetid = datasetid[0]+str(2)+datasetid[2]

    datasetid = int(datasetid)
    pval_datasetid = int(pval_datasetid)

    _exit_err = False 
    
    if datasetid not in dataset_config:
        with open("./get_pvalue_dataset_id.log", "a") as of:
            of.write(f"Dataset set id: {datasetid} not found in dataset config\n")
        _exit_err = True

    if pval_datasetid not in dataset_config:
        with open("./get_pvalue_dataset_id.log", "a") as of:
            of.write(f"P-value dataset set id: {pval_datasetid} not found in dataset config\n")
        _exit_err = True

    if _exit_err:
        print(0)
    else:
        print(pval_datasetid)
