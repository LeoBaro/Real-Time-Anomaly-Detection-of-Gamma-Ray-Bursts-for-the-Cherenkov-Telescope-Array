import os
from pathlib import Path

import pandas as pd 

from sagsci.tools.utils import get_obs_pointing
from sagsci.tools.utils import *
from sagsci.wrappers.rtaph.photometry import Photometrics
from rtapipe.lib.dataset.data_manager import DataManager
from rtapipe.lib.datasource.Photometry3 import OnlinePhotometry 

class LiMa:

    @staticmethod
    def get_li_ma_bins(integration_time, tobs):
        tbins = []
        t_min=0
        while t_min + integration_time <= tobs:
            tbins.append(f"{t_min}-{t_min+integration_time}")
            t_min += integration_time
        return tbins

    @staticmethod
    def detect(pht_list=None, strategy="binned", integration_time=5, stride=5, temporal_bins=[], tobs=500, sigma_gt=0, get_first=False, region_radius=0.2, erange=[0.03, 1]):
        
        if pht_list is None:
            raise ValueError("No pht_list provided")
            
        if strategy not in ["binned", "cumulative"]:
            raise ValueError(f"Unknown Li&Ma strategy: {strategy}")

        pointing = get_obs_pointing(pht_list)
        template = DataManager.extract_runid_from_name(pht_list)
        target = OnlinePhotometry.get_target(Path(os.environ["DATA"]).joinpath("templates", "grb_afterglow", "GammaCatalogV1.0", f"{template}.fits"))
        target['rad'] = region_radius

        phm = Photometrics({'events_filename': pht_list})
        off_regions = Photometrics.find_off_regions('cross', target, pointing, target['rad'])

        data = []
        if strategy == "binned":
            if len(temporal_bins) > 0:
                _temporal_bins = [(float(t.split("-")[0]), float(t.split("-")[1])) for t in temporal_bins]
                for tb in _temporal_bins:
                    on, off, alpha, excess, sigma, err_note = phm.counting(src=target, rad=target['rad'], off_regions=off_regions, e_min=erange[0], e_max=erange[1], t_min=tb[0], t_max=tb[1], draconian=False)
                    data.append([tb[0], tb[1], on, off, alpha, excess, sigma, err_note])
            else:
                t_min = 0
                while t_min + stride <= tobs:
                    on, off, alpha, excess, sigma, err_note = phm.counting(src=target, rad=target['rad'], off_regions=off_regions, e_min=erange[0], e_max=erange[1], t_min=t_min, t_max=t_min + integration_time, draconian=False)
                    data.append([t_min, t_min + integration_time, on, off, alpha, excess, sigma, err_note])
                    t_min += stride

        elif strategy == "cumulative":
            for tmax in range(0, tobs, 5):
                on, off, alpha, excess, sigma, err_note = phm.counting(src=target, rad=target['rad'], off_regions=off_regions, e_min=erange[0], e_max=erange[1], t_min=0, t_max=tmax, draconian=False)
                data.append([0, tmax, on, off, alpha, excess, sigma, err_note])

        df = pd.DataFrame(columns=["tmin", "tmax", "on", "off", "alpha", "excess", "sigma", "err_note"], data=data)
        if sigma_gt > 0:
            df = df[df["sigma"] > sigma_gt]

        if df.shape[0]>0 and get_first:
            return df.iloc[0]

        return df
