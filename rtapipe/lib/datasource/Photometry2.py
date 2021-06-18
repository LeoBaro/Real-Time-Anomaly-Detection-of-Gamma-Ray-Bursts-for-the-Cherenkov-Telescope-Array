import os
import numpy as np
from math import sqrt
from tqdm import tqdm
from pathlib import Path

from RTAscience.cfg.Config import Config
from RTAscience.lib.RTAUtils import get_pointing
from astro.lib.photometry import Photometrics

from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils


class Photometry2:
    """
        This class uses the astro library to integrate gamma like counts in time and energy.
    """
    def __init__(self, dataDir, outputDir):        
        
        """
            dataDir is the simulation directory and it must contain a config.yaml file
        """
        if "DATA" not in os.environ:
            raise EnvironmentError("Please, export $DATA")

        cfg = Config(Path(dataDir).joinpath("config.yaml"))

        # These parameters depend on simulation
        self.runId = cfg.get('runid')
        self.template =  Path(os.environ["DATA"]).joinpath(f'templates/{self.runId}.fits')
        self.pointing = get_pointing(self.template)
        self.simtype = cfg.get('simtype')
        self.onset = cfg.get('onset')
        
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)

        # get files
        self.dataFiles = FileSystemUtils.getAllFiles(dataDir)

    def updatePointing(self):
        pass

    @staticmethod
    def getLinearWindows(wmin, wmax, wsize, wstep):
        if wsize == 0:
            raise ValueError("The 'wsize' argument must be greater than zero.")
        windows = []
        w_start = wmin
        while w_start + wsize <= wmax:
            windows.append((w_start, w_start + wsize))
            w_start += wstep
        return windows   

    @staticmethod
    def getLogWindows(wmin, wmax, howMany):
        windows = []
        npwindows = np.geomspace(wmin, wmax, num=howMany+1)
        for idx,_ in enumerate(npwindows):
            if idx == len(npwindows)-1:
                break
            windows.append((round(npwindows[idx],4) , round(npwindows[idx+1], 4)))
        return windows
    
    def getOutputFilePath(self, inputFilePath):
           
        outputFileNameStr = Path(inputFilePath).with_suffix('').name

        outputFileNameStr += f"_simtype_{self.simtype}_onset_{self.onset}"
        
        return self.outputDir.joinpath(outputFileNameStr).with_suffix(".csv")


    
    def integrate(self, inputFilePath, regionRadius, tWindows=[], eWindows=[], pointing=None):

        if tWindows == None:
            tWindows = [(None, None)]

        if eWindows == None:
            eWindows = [(None, None)]

        phm = Photometrics({ 'events_filename': inputFilePath })

        region = {
            'ra': self.pointing[0],
            'dec': self.pointing[1],
        }        

        outputFilePath = self.getOutputFilePath(inputFilePath)
        totalCounts = 0

        with open(f"{outputFilePath}", "w") as of:
            of.write("TMIN,TMAX,EMIN,EMAX,COUNTS,ERROR\n")    

            for ewin in eWindows:
                print(f"Energy bin: {ewin}")
                for twin in tqdm(tWindows):
                
                    counts = phm.region_counter(region, float(regionRadius), tmin=twin[0], tmax=twin[1], emin=ewin[0], emax=ewin[1])
                    # tcenter = round((twin[1]+twin[0])/2, 4)
                    # ecenter = round((ewin[1]+ewin[0])/2, 4)
                    of.write(f"{twin[0]},{twin[1]},{ewin[0]},{ewin[1]},{counts},{round(sqrt(counts), 4)}\n")
                    totalCounts += counts

        return outputFilePath, totalCounts

    
    
    def integrateAll(self, regionRadius, tWindows=[], eWindows=[], pointing=None, limit=None):
        
        totalCounts = 0
        outputFiles = []

        fileToProcess = self.dataFiles
        if limit:
            fileToProcess = self.dataFiles[:limit]

        for fitsFile in fileToProcess:

            outputFilePath, counts = self.integrate(fitsFile, regionRadius, tWindows, eWindows, pointing)
            
            outputFiles.append(outputFilePath)
        
            totalCounts += counts

        return outputFiles, totalCounts