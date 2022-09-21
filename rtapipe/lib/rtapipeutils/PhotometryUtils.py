import numpy as np

from rtapipe.lib.datasource.integrationstrat.IntegrationStrategies import IntegrationType, TimeIntegration, EnergyIntegration, TimeEnergyIntegration, FullIntegration

class PhotometryUtils:

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

    @staticmethod
    def getIntegrationStrategy(integrationType):

        if integrationType == IntegrationType.TIME:
            return TimeIntegration()

        elif integrationType == IntegrationType.ENERGY:
            return EnergyIntegration()

        elif integrationType == IntegrationType.TIME_ENERGY:
            return TimeEnergyIntegration()

        elif integrationType == IntegrationType.FULL:
            return FullIntegration()

        else:
            raise ValueError(f"Integration type {integrationType} not supported")        