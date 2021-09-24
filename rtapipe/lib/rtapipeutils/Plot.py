from pathlib import Path
import matplotlib.pyplot as plt

class Plot:

    def __init__(self, total_columns):
        
        self.total_columns = total_columns
        self.total_rows = 1 # total_subplots // total_columns
        # self.total_rows += total_subplots % total_columns
        # self.position = range(1, total_subplots + 1)

        self.fig, ax = plt.subplot(1,1)
        self.subplots_indexes = [0]
        self.axes = [ax]
        
    def _add_subplot(self):
        
        if len(self.subplots_indexes) == 1:
            return self.axes[0]
        

        ax = self.fig.add_subplot(self.total_rows,self.total_columns,self.position[self.subplots_count])
        self.subplots_count += 1
        return ax
    
    def add_scatter(self, subplot_idex, x, y, label, color=None):
        ax = self._add_subplot()
        ax.scatter(x, y, label=label, color=color)

    def add_histo(self, x, bins, alpha, label, color=None):
        ax = self._add_subplot()
        ax.hist(x, bins=bins, alpha=alpha, label=label, color=color)
    
    def add_plot(self, y, label, color=None):
        ax = self._add_subplot()
        ax.plot(y, label=label, color=color)
    
    def show(self):
        plt.show()
    
    def saveFig(self, outDir, filename, dpi=300):
        outDir = Path(outDir)
        outDir.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(outDir.joinpath(f"{filename}.png"), dpi=300)

if __name__=='__main__':

    plot = Plot(3,3)

    plot.add_histo([1,1,1,1,2,2,2,2,3,3,3,4,4], 30, 0.7, "test", "red")
    plot.add_scatter(range(13),[1,1,1,1,2,2,2,2,3,3,3,4,4], "test", "blue")
    plot.add_plot([1,1,1,1,2,2,2,2,3,3,3,4,4], "test", "red")

    plot.add_histo([1,1,1,1,2,2,2,2,3,3,3,4,4], 30, 0.7, "test", "red")
    plot.add_scatter(range(13),[1,1,1,1,2,2,2,2,3,3,3,4,4], "test", "blue")
    plot.add_plot([1,1,1,1,2,2,2,2,3,3,3,4,4], "test", "red")

    plot.show()