import os
from pathlib import Path
from functools import partial
from os.path import expandvars
from multiprocessing import Pool

from RTAscience.cfg.Config import Config
from RTAscience.lib.RTAUtils import get_pointing
from RTAscience.aph.utils import ObjectConfig
from RTAscience.aph.photometry import Photometrics
from RTAscience.aph.utils import aeff_eval, ObjectConfig

from rtapipe.lib.rtapipeutils.PhotometryUtils import *
from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils
from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils

class Photometry2:
    """
        This class integrates gamma-like counts in time and energy.
    """
    def __init__(self, configPath, dataDir, outputDir):

        if "DATA" not in os.environ:
            raise EnvironmentError("Please, export $DATA")
        if "CTOOLS" not in os.environ:
            raise EnvironmentError("Please, export $CTOOLS")

        self.cfg = Config(Path(configPath))

        self.dataDir = dataDir

        # irf
        irfPath = Path(expandvars('$CTOOLS')).joinpath("share","caldb","data","cta",self.cfg.get('caldb'),"bcf",self.cfg.get('irf'))
        self.irfFile = irfPath.joinpath(os.listdir(irfPath).pop())
        print(f"irfFile: {self.irfFile}")

        # template
        template =  Path(os.environ["DATA"]).joinpath("templates",f"{self.cfg.get('runid')}.fits")

        # "get_pointing" does not add the offset so it should be renamed in get_target, since it gives
        # the source position
        self.sourcePosition = get_pointing(template) 

        # other parameters
        self.sim_type = self.cfg.get("simtype")
        self.sim_onset = self.cfg.get("onset")
        self.sim_emin = self.cfg.get("emin")
        self.sim_emax = self.cfg.get("emax")
        self.sim_tmin = self.cfg.get("delay")
        self.sim_tmax = self.cfg.get("tobs")

        # Output directories
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)

        self.integrationStrat = None



    def computeRegion(self, regionRadius, offset):
        """ Computes the region ra and dec, starting from the source position (== map center)
            and adding an offset.
        """
        # TODO: the offset should depend on the regionRadius
        return {
            'ra': self.sourcePosition[0] + offset,
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



    def integrateAll(self, integrationType, regionRadius, offset, tWindows=None, eWindows=None, limit=None, parallel=False, procNumber=10, normalize=True, batchSize=1200):
        """
            Integrate multiples input files in a directory. This method is optimized for the processing
            of a large directory of small files (p-value analysis).
        """

        region = self.computeRegion(regionRadius, offset)

        if tWindows is None:
            tWindows = [(self.sim_tmin, self.sim_tmax)]

        if eWindows is None:
            eWindows = [(self.sim_emin, self.sim_emax)]

        areaEffForEnergyBins = None
        if normalize:
            areaEffForEnergyBins = self.computeAeffArea(eWindows, region)
        
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
                print(f"Done! Processed {outputFilesCounts} files.")
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
        self.integrationStrat = PhotometryUtils.getIntegrationStrategy(integrationType)

        outputFileName = Path(inputFilePath).with_suffix('').name + f"_itype_{integrationType}_itime_{tWindows[0][1]-tWindows[0][0]}_normalized_{normalize}.csv"
        outputFilePath = self.outputDir.joinpath(outputFileName)

        if outputFilePath.exists():
            # print(f"Skipped {outputFilePath}. Already exist!")
            return outputFilePath, 0

        photometrics = Photometrics({ 'events_filename': inputFilePath })

        return self.integrationStrat.integrate(photometrics, outputFilePath, region, tWindows, eWindows, parallel=parallel, normalize=normalize, normConfTemplate=None, pointing=None, areaEffForEnergyBins=areaEffForEnergyBins)
