import os
import numpy as np
from pathlib import Path
from functools import partial
from os.path import expandvars
from multiprocessing import Pool

from RTAscience.cfg.Config import Config
from RTAscience.aph.utils import ObjectConfig
from RTAscience.lib.RTAUtils import get_pointing, ObjectConfig, phm_options, aeff_eval
from astro.lib.photometry import Photometrics

from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils
from rtapipe.lib.datasource.integrationstrat.IntegrationStrategies import IntegrationType, TimeIntegration, EnergyIntegration, TimeEnergyIntegration, FullIntegration


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
        self.index = cfg.get('index')
        self.caldb = cfg.get('caldb')
        self.irf = cfg.get('irf')
        self.delay = cfg.get('delay')
        self.tobs = cfg.get('tobs')


        # These parameters are used to normalized the counts
        self.opts = phm_options(erange=[self.emin, self.emax], texp=self.tobs, time_int=[self.delay, self.delay+self.tobs], target=self.pointing, pointing=pointing, index=self.index, save_off_reg=None, irf_file=Path(expandvars('$CTOOLS')).joinpath(f"share/caldb/data/cta/{self.caldb}/bcf/{self.irf}/irf_file.fits"))
        self.conf = ObjectConfig(self.opts)
        self.region_eff_resp = aeff_eval(self.conf, self.region, {'ra': self.pointing[0], 'dec': self.pointing[1]})
        self.livetime = self.opts['end_time'] - self.opts['begin_time']   

        # Output directories 
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)

        # get files
        self.dataFiles = FileSystemUtils.getAllFiles(dataDir)

        self.integrationStrat = None

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
    
    def getOutputFilePath(self, inputFilePath, integrationType):
           
        outputFileNameStr = Path(inputFilePath).with_suffix('').name

        outputFileNameStr += f"_{integrationType}_simtype_{self.simtype}_onset_{self.onset}"
        
        return self.outputDir.joinpath(outputFileNameStr).with_suffix(".csv")

    def setIntegrationType(self, tWindows, eWindows):
        
        self.integrationStrat = None 

        if tWindows and not eWindows:
            self.integrationStrat = TimeIntegration()
            return IntegrationType.TIME

        elif not tWindows and eWindows:
            self.integrationStrat = EnergyIntegration()
            return IntegrationType.ENERGY
        
        elif tWindows and eWindows:
            self.integrationStrat = TimeEnergyIntegration()
            return IntegrationType.TIME_ENERGY
        
        else:
            self.integrationStrat = FullIntegration()
            return IntegrationType.FULL
    
    def integrate(self, inputFilePath, regionRadius, tWindows=None, eWindows=None, pointing=None, parallel=False, normalize=True):
        
        integrationType = self.setIntegrationType(tWindows, eWindows)

        photometrics = Photometrics({ 'events_filename': inputFilePath })

        region = {
            'ra': self.pointing[0],
            'dec': self.pointing[1],
            'rad' : regionRadius
        }        

        outputFilePath = self.getOutputFilePath(inputFilePath, integrationType)
        
        counts = self.integrationStrat.integrate(photometrics, outputFilePath, region, regionRadius, tWindows, eWindows, parallel)

        if normalize:
            return counts / self.region_eff_resp / self.livetime
    
    def integrateAll(self, regionRadius, tWindows=None, eWindows=None, pointing=None, limit=None):
        
        totalCounts = 0
        outputFiles = []

        filesToProcess = self.dataFiles
        if limit:
            filesToProcess = self.dataFiles[:limit]

        func = partial(self.integrate, regionRadius=regionRadius, tWindows=tWindows, eWindows=eWindows, pointing=pointing, parallel=True)

        output = None

        with Pool() as p:

            output = p.map(func, filesToProcess)
        
        # print(output) # [ (PosixPath, counts), (PosixPath, counts), ..]
        
        outputFiles = [str(tuple[0]) for tuple in output]

        totalCounts = sum([tuple[1] for tuple in output])

        return outputFiles, totalCounts
