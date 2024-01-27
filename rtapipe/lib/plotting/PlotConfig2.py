from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class PlotConfig2:

    def __init__(self):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif"
        })        
        plt.rc('font', size=18)          # controls default text sizes
        plt.rc('axes', titlesize=22)     # fontsize of the axes title
        plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
        plt.rc('legend', fontsize=22)    # legend fontsize
        plt.rc('figure', titlesize=24)  # fontsize of the figure title
        #plt.rcParams['legend.title_fontsize'] = 'small'

        # plt.style.use('ieee')

        # Explicitly set the figure size to the desired final figure size. 
        # This ensures that the fonts for the labels will be the right size 
        # compared to the content of the figure, and also ensures that if the 
        # defaults change, your plot will not.
        self.fig_size = (15, 11)
        self.fig_suptitle_size = 35
        self.fig_title_size = 25

        # Choose the font family (‘serif’ or ‘sans-serif’ to match the paper, 
        # most of the time, ‘serif’ is best):
        # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
        # rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':23})

        # Set the font used for MathJax - more on this later
        #rc('mathtext',**{'default':'regular'})

        # Differentiate the font size/style between the axis labels and the tick labels 
        # (by default these are the same, and are set to medium). 
        #rc('xtick', labelsize='x-small')
        #rc('ytick', labelsize='x-small')

        #self.error_kw = {'capsize': 5, 'capthick': 1}

        self.legend_kw = {'size': 17}


        # https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=5
        self.colors = ["#ff7f00", "#377eb8", "#4daf4a", "#984ea3", "#e41a1c"]
        self.markers = ["s", "^", "+", "x"]

        self.dpi = 300
