import os 
from pathlib import Path
if __name__=='__main__':
    current_dir = Path(__file__).absolute().resolve().parent    
    c = 0
    current_dir.joinpath("fits_data").mkdir(parents=True, exist_ok=True)
    for root, folder, files in os.walk(current_dir.joinpath("sim_output")):
        for file in files:
            if file.endswith(".fits"):
                source = os.path.join(root, file)
                dest = os.path.join(Path(root).parent.parent, "fits_data", file)
                #os.rename(os.path.join(root, file), os.path.join(root, file.replace("sim_output", "simulated_photon_lists")))  
                # copy files
                os.system(f"cp {source} {dest}")
                c += 1
    print("Total copied files: ", c)