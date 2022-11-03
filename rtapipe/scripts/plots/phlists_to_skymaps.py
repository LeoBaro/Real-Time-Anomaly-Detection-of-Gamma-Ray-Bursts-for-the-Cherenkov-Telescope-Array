import os
import argparse
from pathlib import Path
from RTAscience.cfg.Config import Config
from sagsci.tools.plotting import SkyImage
from RTAscience.lib.RTAUtils import get_pointing
from rtapipe.lib.datasource.Photometry3 import OnlinePhotometry

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--conf-file', type=str, required=True)    
    parser.add_argument('--subtraction', type=str, choices=["NONE","IRF"], required=True)
    args = parser.parse_args()

    cfg = Config(args.conf_file)

    #rta.caldb = cfg.get('caldb')
    #rta.irf = cfg.get('irf')

    template =  Path(os.environ["DATA"]).joinpath("templates",f"{cfg.get('runid')}.fits")
    target = get_pointing(template) 
    
    pht = OnlinePhotometry(args.conf_file)
    regions_conf = pht.create_photometry_configuration(region_radius=0.2, number_of_energy_bins=3, rings_n=1, flatten=True)
    regions = [r[0] for r in regions_conf]
    # print(regions)

    for root, subdirs, files in os.walk(args.data_dir):

        fits_files = [f for f in files if ".fits" in f and ".skymap" not in f]

        for filename in fits_files: 

            plot = SkyImage()
            plot.set_target(ra=target[0], dec=target[1])
            plot.set_pointing(ra=target[0], dec=target[1])
            img = os.path.join(root, filename)
            out = img.replace('.fits', '.png')
            print(f"Producing.. {out}")
            plot.counts_map_with_regions(
                    img, 
                    regions, 
                    trange=None, 
                    erange=[cfg.get('emin'), cfg.get('emax')], 
                    roi=cfg.get('roi'), 
                    name=out, 
                    title="Skymap", 
                    )
            del plot
        

if __name__=="__main__":
    main()
