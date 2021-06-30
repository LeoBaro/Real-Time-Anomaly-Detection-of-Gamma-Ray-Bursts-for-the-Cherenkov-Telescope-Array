from math import sqrt
from tqdm import tqdm
from abc import ABC, abstractmethod

class IntegrationType:
    TIME = "t"
    ENERGY = "e"
    TIME_ENERGY = "te"
    FULL = "full"
    
class IntegrationStrategy(ABC):
    @abstractmethod
    def integrate(self, photometrics, outputFilePath, region, regionRadius, tWindows, eWindows):
        pass

class TimeIntegration(ABC):

    def integrate(self, photometrics, outputFilePath, region, regionRadius, tWindows, eWindows, parallel=False):

        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write("TMIN,TMAX,COUNTS,ERROR\n")    
            for twin in tqdm(tWindows, disable=parallel):
                counts = photometrics.region_counter(region, float(regionRadius), tmin=twin[0], tmax=twin[1], emin=None, emax=None)
                of.write(f"{twin[0]},{twin[1]},{counts},{round(sqrt(counts), 4)}\n")
                totalCounts += counts

        return outputFilePath, totalCounts

class EnergyIntegration(ABC):

    def integrate(self, photometrics, outputFilePath, region, regionRadius, tWindows, eWindows, parallel):

        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write("EMIN,EMAX,COUNTS,ERROR\n")    
            for ewin in tqdm(eWindows, disable=parallel):
                counts = photometrics.region_counter(region, float(regionRadius), tmin=None, tmax=None, emin=ewin[0], emax=ewin[1])
                of.write(f"{ewin[0]},{ewin[1]},{counts},{round(sqrt(counts), 4)}\n")
                totalCounts += counts

        return outputFilePath, totalCounts

class TimeEnergyIntegration(ABC):
    def integrate(self, photometrics, outputFilePath, region, regionRadius, tWindows, eWindows, parallel=False):
                
        header="TMIN,TMAX"
        for energyBin in eWindows:
            header += f",COUNTS_{energyBin[0]}-{energyBin[1]},ERROR_{energyBin[0]}-{energyBin[1]}"
        header += "\n"

        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write(header)    
            for twin in tqdm(tWindows, disable=parallel):
                of.write(f"{twin[0]},{twin[1]}")                
                for ewin in eWindows:
                    counts = photometrics.region_counter(region, float(regionRadius), tmin=twin[0], tmax=twin[1], emin=ewin[0], emax=ewin[1])
                    of.write(f",{counts},{round(sqrt(counts), 4)}")
                    totalCounts += counts

                of.write("\n")

        return outputFilePath, totalCounts

class FullIntegration(ABC):
    def integrate(self, photometrics, outputFilePath, region, regionRadius, tWindows, eWindows, parallel=False):
        
        totalCounts = 0
        with open(f"{outputFilePath}", "w") as of:
            of.write("COUNTS,ERROR\n")    
            counts = photometrics.region_counter(region, float(regionRadius), tmin=None, tmax=None, emin=None, emax=None)
            of.write(f"{counts},{round(sqrt(counts), 4)}\n")
            totalCounts += counts

        return outputFilePath, totalCounts