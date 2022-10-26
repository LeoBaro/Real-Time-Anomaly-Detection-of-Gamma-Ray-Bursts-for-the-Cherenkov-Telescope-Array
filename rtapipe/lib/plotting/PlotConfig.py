from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class PlotConfig:

    def __init__(self):
        plt.style.use('ieee')

        # Explicitly set the figure size to the desired final figure size. 
        # This ensures that the fonts for the labels will be the right size 
        # compared to the content of the figure, and also ensures that if the 
        # defaults change, your plot will not.
        self.fig_size = (15, 8)
        self.fig_suptitle_size = 20

        # Choose the font family (‘serif’ or ‘sans-serif’ to match the paper, 
        # most of the time, ‘serif’ is best):
        # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
        rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':15})

        # Set the font used for MathJax - more on this later
        rc('mathtext',**{'default':'regular'})

        # Differentiate the font size/style between the axis labels and the tick labels 
        # (by default these are the same, and are set to medium). 
        rc('xtick', labelsize='x-small')
        rc('ytick', labelsize='x-small')

        self.error_kw = {'capsize': 5, 'capthick': 1}

        self.legend_kw = {'size': 15}

        # https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=5
        self.colors = ["#ff7f00", "#377eb8", "#4daf4a", "#984ea3", "#e41a1c"]
        self.markers = ["s", "^", "+", "x"]

        self.dpi = 300

    def test(self):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot([1, 2, 3, 4])
        ax.set_xlabel('The x values')
        ax.set_ylabel('The y values')
        plt.savefig('test2.png', dpi=300)

if __name__=='__main__':
    PlotConfig().test()
