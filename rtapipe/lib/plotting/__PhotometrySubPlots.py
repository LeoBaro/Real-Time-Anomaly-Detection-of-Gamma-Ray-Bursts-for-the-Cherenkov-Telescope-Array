import matplotlib.pyplot as plt

from rtapipe.lib.plotting.PhotometryPlot import PhotometryPlot

class PhotometrySubPlots(PhotometryPlot):

    def __init__(self, title=None):
        super().__init__(title)
        self.FM, self.FMX  = plt.subplots(2, 2)
        self.FM.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        self.current_axis_idx = 0
        self.axes = [(0,0), (0,1), (1,0), (1,1)]

    def nextAxis(self):
        self.current_axis_idx = (self.current_axis_idx + 1) % len(self.axes)
        self.ccount = (self.ccount + 1) % len(self.axes)
        x,y = self.axes[self.current_axis_idx]
        return self.FMX[x][y]
    
    def prevAxis(self):
        self.current_axis_idx = (self.current_axis_idx - 1) % len(self.axes)
        self.ccount = (self.ccount - 1) % len(self.axes)
        x,y = self.axes[self.current_axis_idx]
        return self.FMX[x][y]

    def currentAxis(self):
        x,y = self.axes[self.current_axis_idx]     
        return self.FMX[x][y]

    def addData(self, photometry_csv_file, args, label_on, integration, vertical_line=False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
        data = super().getData(photometry_csv_file)

        if as_baseline:
            axis = self.prevAxis()
        else:
            axis = self.currentAxis()
            
        label_on_string = self.getLabel(label_on, args)

        window_size = data["valmax"] - data["valmin"]

        if integration == "t":

            if not as_baseline: 
                plotargs = {"color": PhotometryPlot.colors[self.ccount]}
            else:
                plotargs = {"color": baseline_color}

            axis.scatter(data["x"], data["y"], label = label_on_string, **plotargs, s=0.1)
            axis.errorbar(data["x"], data["y"], xerr=window_size, yerr=data["err"], fmt="o", **plotargs)

            if not as_baseline:
                if vertical_line:
                    axis.axvline(x=vertical_line_x, color="red", linestyle="--")


        elif integration == "e":
            plotargs = {
                "yerr": data["err"], 
                "label": label_on_string,
                "alpha": 0.2, 
                "width": args["e_window_step"]
            }
            if as_baseline:
                plotargs["linewidth"] = 4
                plotargs["color"] = "black"
            else:
                plotargs["color"] = PhotometryPlot.colors[self.ccount]

            _ = axis.bar(data["x"], data["y"], **plotargs)
            

        elif integration == "et":
            pass

        elif integration == "te":
            pass


        self.FM.suptitle(self.getTitle(label_on, args))
        # axis.set_title(label_on_string)
        axis.set_ylabel('Counts')
        axis.set_xlabel(f'Window center (integration: "{integration}")')

        axis.legend(loc="best")

        self.nextAxis()
        # self.FM.tight_layout(h_pad=1.5)
        # self.FM.tight_layout()

    def save(self, outfileName):
        outfileName = outfileName+'.png'
        self.FM.savefig(outfileName)
        print("Produced: ", outfileName)

    def show(self):
        plt.show()