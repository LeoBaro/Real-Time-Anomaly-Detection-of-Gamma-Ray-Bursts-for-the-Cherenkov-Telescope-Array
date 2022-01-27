import os
from unicodedata import normalize
import numpy as np
from pathlib import Path
from functools import partial
from os.path import expandvars
from multiprocessing import Pool

from RTAscience.cfg.Config import Config
from RTAscience.lib.RTAUtils import get_pointing
from RTAscience.aph.utils import ObjectConfig
from astro.lib.photometry import Photometrics
from RTAscience.aph.utils import aeff_eval, ObjectConfig

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
        self.irfFile = Path(expandvars('$CTOOLS')).joinpath(f"share/caldb/data/cta/{cfg.get('caldb')}/bcf/{cfg.get('irf')}/irf_file.fits")
        template =  Path(os.environ["DATA"]).joinpath(f'templates/{cfg.get("runid")}.fits')

        self.sourcePosition = get_pointing(template) # alias get_target
        self.sim_type = cfg.get("simtype")
        self.sim_onset = cfg.get("onset")
        self.sim_emin = cfg.get("emin")
        self.sim_emax = cfg.get("emax")
        self.sim_tmin = cfg.get("delay")
        self.sim_tmax = cfg.get("tobs")

        # Output directories
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)

        self.dataDir = dataDir
        # get files
        # self.dataFiles = FileSystemUtils.getAllFiles(dataDir)

        self.integrationStrat = None

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

    def computeRegion(self, regionRadius, regionOffset = 2):
        """ Computes the region ra and dec, starting from the source position (== map center)
            and adding an offset.
        """
        # TODO: the offset should depend on the regionRadius
        return {
            'ra': self.sourcePosition[0] + regionOffset,
            'dec': self.sourcePosition[1],
            'rad' : float(regionRadius)
        }

    def computeAeffArea(self, eWindows, region):

        # This configuration is needed by Ambra's normalization tool
        # The None parameters will be updated later
        normConfTemplate = ObjectConfig({
            "begin_time" : 0,
            "end_time" : 0,
            "source_ra" : self.sourcePosition[0],
            "source_dec" : self.sourcePosition[1],
            "region_radius" : region["rad"],
            "verbose" : 0,
            "energy_min" : 0,
            "energy_max" : 0,
            "pixel_size" : 0.05,
            "power_law_index" : -2.1,
            "irf_file" : self.irfFile
        })

        # I dati sono stati generati tenendo conto del puntamento del telescopio (e la posizione della sorgente)
        # Quindi devo tenere in considerazione l'offaxis per la normalizzazione:
        #    - se l'offaxis angle non cambia (il telescopio si muove insieme alla sorgente) allora aeff_eval non cambia
        #    - se il telescopio non segue la sorgente, cambia l'offaxis, cambia l'area efficace

        # pointing => quando il telescopio segue la sorgente il pointing è equivalente alla posizione della sorgente
        # altrimenti si può prendere anche dall'header del FITS  --> TODO <--
        pointing = self.sourcePosition

        areaEffForEB = {}
        for ewin in eWindows:
            normConfTemplate.energy_min = ewin[0]
            normConfTemplate.energy_max = ewin[1]
            areaEffForEB[str(ewin)] = aeff_eval(normConfTemplate, region, {'ra': pointing[0], 'dec': pointing[1]})

        return areaEffForEB

    def getOutputFilePath(self, inputFilePath, integrationType, normalize):

        outputFileNameStr = Path(inputFilePath).with_suffix('').name

        outputFileNameStr += f"_{integrationType}_simtype_{self.sim_type}_onset_{self.sim_onset}_normalized_{normalize}"

        return self.outputDir.joinpath(outputFileNameStr).with_suffix(".csv")

    def setIntegrationType(self, integrationType):

        self.integrationStrat = None

        if integrationType == IntegrationType.TIME:
            self.integrationStrat = TimeIntegration()
            return IntegrationType.TIME

        elif integrationType == IntegrationType.ENERGY:
            self.integrationStrat = EnergyIntegration()
            return IntegrationType.ENERGY

        elif integrationType == IntegrationType.TIME_ENERGY:
            self.integrationStrat = TimeEnergyIntegration()
            return IntegrationType.TIME_ENERGY

        elif integrationType == IntegrationType.FULL:
            self.integrationStrat = FullIntegration()
            return IntegrationType.FULL

        else:
            raise ValueError(f"Integration type {integrationType} not supported")



    def integrateAll(self, integrationType, regionRadius, tWindows=None, eWindows=None, limit=None, parallel=False, procNumber=10, normalize=True, batchSize=1200):
        """
            Integrate multiples input files in a directory
        """
        region = self.computeRegion(regionRadius)

        if tWindows is None:
            tWindows = [(self.sim_tmin, self.sim_tmax)]

        if eWindows is None:
            eWindows = [(self.sim_emin, self.sim_emax)]

        areaEffForEnergyBins = None
        if normalize:
            areaEffForEnergyBins = self.computeAeffArea(eWindows, region)
            
        if integrationType == "T":
            integrationType = self.setIntegrationType(IntegrationType.TIME)

        #elif integrationType == "E":
        #    integrationType = self.setIntegrationType(IntegrationType.ENERGY)

        elif integrationType == "TE":
            integrationType = self.setIntegrationType(IntegrationType.TIME_ENERGY)

        elif integrationType == "F":
            integrationType = self.setIntegrationType(IntegrationType.FULL)

        else:
            raise ValueError(f"Integration {integrationType} is not supported")

        func = partial(self.integrate, integrationType=integrationType, region=region, tWindows=tWindows, eWindows=eWindows, areaEffForEnergyBins=areaEffForEnergyBins, parallel=parallel, normalize=normalize)

        output = None
        totalCounts = 0
        outputFilesCounts = 0
        batchIterator = FileSystemUtils.iterDirBatch(self.dataDir, batchSize)

        while True:
            try:
                batchFiles = next(batchIterator)
                with Pool(procNumber) as p:
                    output = p.map(func, batchFiles)
                    outputFilesCounts += len([str(tuple[0]) for tuple in output])
                    totalCounts = sum([tuple[1] for tuple in output])

            except StopIteration:
                print(f"Processed {outputFilesCounts} files")
                break

        del batchIterator

        return outputFilesCounts, totalCounts


    def integrate(self, inputFilePath, integrationType, region, tWindows, eWindows, parallel=False, normalize=True, areaEffForEnergyBins=None):
        """
            Può usare la posizione della sorgente (self.sourcePosition) (che sta nell'header del template).
            Oppure, può usare una positione custom passata come parametro. Oppure un offest??

            Params:
                inputFilePath: str ->
                outputFilePath: str ->
                region: tuple -> The region that describes the spatial integration. If None, the region (ra,dec) will correspond to the source position.
                tWindows: list ->
                eWindows: list ->
        """

        outputFilePath = self.getOutputFilePath(inputFilePath, integrationType, normalize)

        if Path(outputFilePath).exists():
            # print(f"Skipped {outputFilePath}. Already exist!")
            return outputFilePath, 0

        photometrics = Photometrics({ 'events_filename': inputFilePath })

        return self.integrationStrat.integrate(photometrics, outputFilePath, region, tWindows, eWindows, parallel=parallel, normalize=normalize, normConfTemplate=None, pointing=None, areaEffForEnergyBins=areaEffForEnergyBins)
