import os         
from os import listdir
from os.path import isfile, join
from pathlib import Path

from RTAscience.cfg.Config import Config
from RTAscience.lib.RTAUtils import get_pointing

def getInput(dataFolder, outdir, howMany=-1):

    datapath = Path("/data01/home/baroncelli/phd/DATA")

    os.environ["DATA"] = str(datapath)

    cfg = Config(dataFolder.joinpath("config.yaml"))
    
    # get pointing from template
    runid = cfg.get('runid')
    template =  os.path.join(datapath, f'templates/{runid}.fits')
    pointing = get_pointing(template)
    
    # get files
    datafiles = [join(dataFolder, f) for f in listdir(dataFolder) 
                    if isfile(join(dataFolder, f)) and ".yaml" not in f and ".log" not in f and "IRF" not in f] 
    
    # get simulation parameters
    sim_params = []
    for datafile in datafiles:
        
        conf = {
            'input_file': str(datafile),
            'output_dir': str(outdir),
            'simtype' : cfg.get('simtype'),
            'runid' : cfg.get('runid'),
            't_window_start': 0,
            't_window_stop': cfg.get('tobs'),
            'e_window_start': cfg.get('emin'),
            'e_window_stop': cfg.get('emax'),
            'onset' : cfg.get('onset')
        }
        sim_params.append(conf)

        if len(sim_params) >= howMany:
            break
            
            
    print(f"Found: {len(sim_params)} files, pointing is: {pointing}")

    return sim_params, pointing