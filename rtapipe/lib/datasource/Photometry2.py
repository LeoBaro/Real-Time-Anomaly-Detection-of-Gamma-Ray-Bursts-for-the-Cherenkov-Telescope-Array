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

        self.sourcePosition = get_pointing(template) # source position
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

    def computeAeffArea(self, eWindows, regionRadius, region=None):

        # for the moment I dont need phm_options
        # opts = phm_options(erange=[emin, emax], texp=(tmax-tmin), time_int=(tmin, tmax), target=normParams.pointing, pointing=normParams.pointing, index=normParams.index, save_off_reg=None, irf_file=Path(expandvars('$CTOOLS')).joinpath(f"share/caldb/data/cta/{normParams.caldb}/bcf/{normParams.irf}/irf_file.fits"))

        # This configuration is needed by Simone's tool.
        # It describe the spatial integration.
        reg = {
            'ra': None,
            'dec': None,
            'rad' : float(regionRadius)
        }
        # You can integrate in a region by specifying the region input parameter
        # If region param is None, the region will be defined by the source position.
        if region is not None:
            reg["ra"] = region[0]
            reg["dec"] = region[1]
        else:
            reg["ra"] = self.sourcePosition[0]
            reg["dec"] = self.sourcePosition[1]

        # This configuration is needed by Ambra's normalization tool
        # The None parameters will be updated later
        normConfTemplate = ObjectConfig({
            "begin_time" : 0,
            "end_time" : 0,
            "source_ra" : self.sourcePosition[0],
            "source_dec" : self.sourcePosition[1],
            "region_radius" : regionRadius,
            "verbose" : 0,
            "energy_min" : 0,
            "energy_max" : 0,
            "pixel_size" : 0.05,
            "power_law_index" : -2.4,
            "irf_file" : self.irfFile
        })

        # I dati sono stati generati tenendo conto del puntamento del telescopio (e la posizione della sorgente)
        # Quindi devo tenere in considerazione l'offaxis per la normalizzazione:
        #    - se l'offaxis angle non cambia (il telescopio si muove insieme alla sorgente) allora aeff_eval non cambia
        #    - se il telescopio non segue la sorgente, cambia l'offaxis, cambia l'area efficace

        # pointing => quando il telescopio segue la sorgente il pointing è equivalente alla posizione della sorgente
        # altrimenti si può prendere anche dall'header del FITS  --> TODO <--
        pointing = self.sourcePosition

        regionEffForEB = {}
        for ewin in eWindows:
            normConfTemplate.energy_min = ewin[0]
            normConfTemplate.energy_max = ewin[1]
            regionEffForEB[str(ewin)] = aeff_eval(normConfTemplate, reg, {'ra': pointing[0], 'dec': pointing[1]})

        return reg, regionEffForEB



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



    def integrateAll(self, integrationType, region, regionRadius, tWindows, eWindows, limit=None, parallel=False, procNumber=10, normalize=True):
        """
            Integrate multiples input files in a directory
        """
        totalCounts = 0
        outputFiles = []

        if integrationType == "T":
            func = partial(self.integrateT, region=region, regionRadius=regionRadius, tWindows=tWindows, parallel=parallel, normalize=normalize)
        elif integrationType == "E":
            func = partial(self.integrateE, region=region, regionRadius=regionRadius, eWindows=eWindows, parallel=parallel, normalize=normalize)
        elif integrationType == "TE":
            func = partial(self.integrateTE, region=region, regionRadius=regionRadius, tWindows=tWindows, eWindows=eWindows, parallel=parallel, normalize=normalize)
        elif integrationType == "F":
            func = partial(self.integrateF, region=region, regionRadius=regionRadius, parallel=parallel, normalize=normalize)

        output = None
        count = 0
        batchIterator = FileSystemUtils.iterDirBatch(self.dataDir, 1200)

        # OPTIMIZE: pass multiple files to integrate..

        while True:
            try:
                batchFiles = next(batchIterator)
                with Pool(procNumber) as p:
                    output = p.map(func, batchFiles)
                    # print(output) # [ (PosixPath, counts), (PosixPath, counts), ..]
                    outputFiles = [str(tuple[0]) for tuple in output]
                    totalCounts = sum([tuple[1] for tuple in output])
                count += len(batchFiles)
            except StopIteration:
                print(f"Processed {count} files")
                break

        del batchIterator

        return outputFiles, totalCounts


    """
        API
    """
    def integrateT(self, inputFilePath, region, regionRadius, tWindows, parallel=False, normalize=True):
        """
            Integrate along temporal and spatial dimensions.
        """
        integrationType = self.setIntegrationType(IntegrationType.TIME)
        outputFilePath = self.getOutputFilePath(inputFilePath, integrationType, normalize)
        eWindows = [(self.sim_emin, self.sim_emax)]
        region, regionEffForEB = self.computeAeffArea(eWindows, regionRadius, region)
        return self.integrate(inputFilePath, outputFilePath, region, regionRadius, tWindows, eWindows, parallel, normalize, regionEffForEB)

    def integrateE(self, inputFilePath, region, regionRadius, eWindows, parallel=False, normalize=True):
        """
            Integrate along energetic and spatial dimensions.
        """
        integrationType = self.setIntegrationType(IntegrationType.ENERGY)
        outputFilePath = self.getOutputFilePath(inputFilePath, integrationType, normalize)
        tWindows = [(self.sim_tmin, self.sim_tmax)]
        region, regionEffForEB = self.computeAeffArea(eWindows, regionRadius, region)
        return self.integrate(inputFilePath, outputFilePath, region, regionRadius, tWindows, eWindows, parallel, normalize, regionEffForEB)

    def integrateTE(self, inputFilePath, region, regionRadius, tWindows, eWindows, parallel=False, normalize=True):
        """
            Integrate along temporal, energetic and spatial dimensions.
        """
        integrationType = self.setIntegrationType(IntegrationType.TIME_ENERGY)
        outputFilePath = self.getOutputFilePath(inputFilePath, integrationType, normalize)
        region, regionEffForEB = self.computeAeffArea(eWindows, regionRadius, region)
        return self.integrate(inputFilePath, outputFilePath, region, regionRadius, tWindows, eWindows, parallel, normalize, regionEffForEB)

    def integrateF(self, inputFilePath, region, regionRadius, parallel=False, normalize=True):
        """
            Integrate only along the spatial dimension.
        """
        integrationType = self.setIntegrationType(IntegrationType.FULL)
        outputFilePath = self.getOutputFilePath(inputFilePath, integrationType, normalize)
        tWindows = [(self.sim_tmin, self.sim_tmax)]
        eWindows = [(self.sim_emin, self.sim_emax)]
        region, regionEffForEB = self.computeAeffArea(eWindows, regionRadius, region)
        return self.integrate(inputFilePath, outputFilePath, region, regionRadius, tWindows, eWindows, parallel, normalize, regionEffForEB)



    def integrate(self, inputFilePath, outputFilePath, region, regionRadius, tWindows, eWindows, parallel=False, normalize=True, regionEffForEB=None):
        """
            Può usare la posizione della sorgente (self.sourcePosition) (che sta nell'header del template).
            Oppure, può usare una positione custom passata come parametro. Oppure un offest??

            Params:
                inputFilePath: str ->
                outputFilePath: str ->
                region: tuple -> The region that describes the spatial integration. If None, the region (ra,dec) will correspond to the source position.
                regionRadius: float -> The radius of the region
                tWindows: list ->
                eWindows: list ->
        """
        if Path(outputFilePath).exists():
            # print(f"Skipped {outputFilePath}. Already exist!")
            return outputFilePath, 0

        photometrics = Photometrics({ 'events_filename': inputFilePath })

        return self.integrationStrat.integrate(photometrics, outputFilePath, region, tWindows, eWindows, parallel=parallel, normalize=normalize, normConfTemplate=None, pointing=None, regionEffForEB=regionEffForEB)
