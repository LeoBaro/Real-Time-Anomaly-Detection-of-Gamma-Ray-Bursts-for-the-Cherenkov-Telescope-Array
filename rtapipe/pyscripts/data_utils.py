import os 
from pathlib import Path
from astropy.io import fits

class DataUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def fitsInfo(fitsFile):
        hdul = fits.open(fitsFile)
        hdul.info()
        print(hdul[1].columns)

    @staticmethod
    def getTheoreticalEmissionModel(templateID):
        dataDir = os.environ.get("DATA")
        if dataDir is None:
            raise ValueError("Please, export DATA=..")
        templateID = Path(dataDir).joinpath("templates","grb_afterglow","GammaCatalogV1.0",f"{templateID}_ebl.fits")

        print(templateID)
        assert True == templateID.is_file()

        hdul = fits.open(templateID)
        # hdul.info()

        # extract model
        energies = (hdul[1].columns, hdul[1].data)
        times = (hdul[2].columns, hdul[2].data)
        spectra = (hdul[3].columns, hdul[3].data)

        return energies, times, spectra

        




# (energies, times, spectra) = DataUtils.getTheoreticalEmissionModel("run0406_ID000126")

"""
print("energies", energies[0])
print("times", times[0])
print("spectra", spectra[0])

print("times", times[1][0])
print("times", times[1].shape)
print("energies", energies[1].shape)
print("spectra", spectra[1].shape)
print("spectra", spectra[1][0][0])
"""

# DataUtils.fitsInfo("/data01/home/baroncelli/phd/DATA/obs/obs_st_bkg_tr_1_os_0_emin_0.03_emax_0.15_roi_2.5/backgrounds/bkg000001.fits")
