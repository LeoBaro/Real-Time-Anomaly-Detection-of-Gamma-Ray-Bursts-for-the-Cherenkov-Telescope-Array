from math import sqrt
from tqdm import tqdm
from pathlib import Path
from os.path import expandvars
from abc import ABC, abstractmethod
from time import time 

from RTAscience.aph.utils import aeff_eval, ObjectConfig
from RTAscience.lib.RTAUtils import get_pointing, phm_options

class IntegrationType:
    TIME = "t"
    ENERGY = "e"
    TIME_ENERGY = "te"
    FULL = "full"
    
class IntegrationStrategy(ABC):
    @abstractmethod
    def integrate(self, photometrics, outputFilePath, region, tWindows, eWindows, parallel=False, normalize=True, normConfTemplate=None, pointing=None):
        pass    

    def computeAeffArea(self, normConfTemplate, emin, emax, region, pointing):
        normConfTemplate.energy_min = emin
        normConfTemplate.energy_max = emax
        region_eff_resp = aeff_eval(normConfTemplate, region, {'ra': pointing[0], 'dec': pointing[1]})
        return region_eff_resp

    def normalize(self, counts, region_eff_resp, livetime):

        normCounts = counts / region_eff_resp / livetime        

        return normCounts


class TimeIntegration(IntegrationStrategy):

    def integrate(self, photometrics, outputFilePath, region, tWindows, eWindows, parallel=False, normalize=True, normConfTemplate=None, pointing=None):
        
        if normalize and (normConfTemplate is None or pointing is None):
            raise ValueError(f"normalize is {normalize} and normConfTemplate is {normConfTemplate} and pointing is {pointing}")

        assert len(eWindows) == 1

        emin = eWindows[0][0]
        emax = eWindows[0][1]

        # Normalization
        livetime = tWindows[0][1] - tWindows[0][0]
        region_eff_resp = self.computeAeffArea(normConfTemplate, emin, emax, region, pointing)

        totalCounts = 0

        with open(f"{outputFilePath}", "w") as of:
        
            of.write("TMIN,TMAX,COUNTS,ERROR\n")    
            
            for twin in tqdm(tWindows, disable=parallel):  

                counts = photometrics.region_counter(region, float(region["rad"]), tmin=twin[0], tmax=twin[1], emin=emin, emax=emax)

                if normalize:

                    counts = self.normalize(counts, region_eff_resp, livetime)

                of.write(f"{twin[0]},{twin[1]},{counts},{round(sqrt(counts), 4)}\n")           
                totalCounts += counts

        return outputFilePath, totalCounts



class EnergyIntegration(IntegrationStrategy):

    def integrate(self, photometrics, outputFilePath, region, tWindows, eWindows, parallel=False, normalize=True, normConfTemplate=None, pointing=None):

        if normalize and (normConfTemplate is None or pointing is None):
            raise ValueError(f"normalize is {normalize} and normConfTemplate is {normConfTemplate} and pointing is {pointing}")

        assert len(tWindows) == 1

        tmin = tWindows[0][0]
        tmax = tWindows[0][1]

        # Normalization
        livetime = tmax - tmin


        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write("EMIN,EMAX,COUNTS,ERROR\n")    

            for ewin in tqdm(eWindows, disable=parallel):

                counts = photometrics.region_counter(region, float(region["rad"]), tmin=None, tmax=None, emin=ewin[0], emax=ewin[1])

                if normalize:

                    region_eff_resp = self.computeAeffArea(normConfTemplate, ewin[0], ewin[1], region, pointing)

                    counts = self.normalize(counts, region_eff_resp, livetime)

                of.write(f"{ewin[0]},{ewin[1]},{counts},{round(sqrt(counts), 4)}\n")
                totalCounts += counts

        return outputFilePath, totalCounts



class TimeEnergyIntegration(IntegrationStrategy):

    def integrate(self, photometrics, outputFilePath, region, tWindows, eWindows, parallel=False, normalize=True, normConfTemplate=None, pointing=None):

        if normalize and (normConfTemplate is None or pointing is None):
            raise ValueError(f"normalize is {normalize} and normConfTemplate is {normConfTemplate} and pointing is {pointing}")

        header="TMIN,TMAX"
        for energyBin in eWindows:
            header += f",COUNTS_{energyBin[0]}-{energyBin[1]},ERROR_{energyBin[0]}-{energyBin[1]}"
        header += "\n"

        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write(header)    
        
            livetime = tWindows[0][1] - tWindows[0][0]
            
            # Compute aeff area for the energy bins
            region_eff_resp_for_eb = {}
            for ewin in tqdm(eWindows, disable=parallel):
                region_eff_resp_for_eb[str(ewin)] = self.computeAeffArea(normConfTemplate, ewin[0], ewin[1], region, pointing)

            for twin in tqdm(tWindows, disable=parallel, leave=False):

                of.write(f"{twin[0]},{twin[1]}")                

                for ewin in eWindows:
                    
                    counts = photometrics.region_counter(region, float(region["rad"]), tmin=twin[0], tmax=twin[1], emin=ewin[0], emax=ewin[1])

                    if normalize:

                        counts = self.normalize(counts, region_eff_resp_for_eb[str(ewin)], livetime)

            
                    of.write(f",{counts},{round(sqrt(counts), 4)}")
                    totalCounts += counts

                of.write("\n")

        return outputFilePath, totalCounts











class FullIntegration(IntegrationStrategy):

    def integrate(self, photometrics, outputFilePath, region, tWindows, eWindows, parallel=False, normalize=True, normConfTemplate=None, pointing=None):
        
        if normalize and (normConfTemplate is None or pointing is None):
            raise ValueError(f"normalize is {normalize} and normConfTemplate is {normConfTemplate} and pointing is {pointing}")

        assert len(tWindows) == 1
        assert len(eWindows) == 1

        tmin = tWindows[0][0]
        tmax = tWindows[0][1]
        emin = eWindows[0][0]
        emax = eWindows[0][1]        

        # Normalization
        livetime = tmax - tmin
        region_eff_resp = self.computeAeffArea(normConfTemplate, emin, emax, region, pointing)
        
        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write("COUNTS,ERROR\n")    

            counts = photometrics.region_counter(region, float(region["rad"]), tmin=None, tmax=None, emin=None, emax=None)

            if normalize:

                counts = self.normalize(counts, region_eff_resp, livetime)

            of.write(f"{counts},{round(sqrt(counts), 4)}\n")
            totalCounts += counts

        return outputFilePath, totalCounts