import numpy as np
import pandas as pd
from math import floor
from pathlib import Path
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
COLORS = list(mcolors.BASE_COLORS)
plt.rcParams.update({'font.size': 18, 'lines.markersize': 0.5,'legend.markerscale': 3, 'lines.linewidth':1, 'lines.linestyle':'-'})
FIG_SIZE = (15,7)
DPI=300

class AnalysisPlot:

    def __init__(self):
        pass


    def plotTrainingTime(self, data):
        
