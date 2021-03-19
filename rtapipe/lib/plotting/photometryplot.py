import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 7, 'lines.markersize': 0.5,'legend.markerscale': 5, 'lines.linewidth':0.5, 'lines.linestyle':'--'})


class PhotometryPlot():
    
    inch_x = 10
    inch_y = 6
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:pink","tab:olive","tab:cyan"]
    
    def __init__(self, title=None):
        self.outPlot = None
        self.title = title
        self.ccount = 0


    def getData(self, photometry_csv_file, integration):
        df = pd.read_csv(photometry_csv_file)
        return {
            "x": df['VALCENTER'],
            "y": df['COUNTS'],
            "err": df['ERROR'],
            "valmax": df['VALMAX'],
            "valmin": df['VALMIN']
        }


    def getColor(self):
        col = PhotometryPlot.colors[self.ccount]
        self.ccount += 1
        return col

    def getForbidden(self):
        return ["datapath", "simtype", "runid", "input_file", "override", "plot"]

    def addData(self, photometry_csv_file, args, sim_params, label_on, integration, as_baseline=False):
        pass

    def show(self):
        pass
    
    def getTitle(self, label_on, args):
        forbidden = self.getForbidden() + label_on
        ok = ["pointing", "region_radius", "t_window_start", "t_window_size", "t_window_step", "t_window_stop", "e_window_start", "e_window_size", "e_window_step", "e_window_stop", "onset"]
        args = [f"{key}: {value} " for key, value in args.items() if key not in forbidden and key in ok]
        mid_point = int(len(list(args))/2)
        if self.title:
            title = self.title + '\n'
        else:
            title = 'Aperture photometry plot\n'
        title += str(args[:mid_point]) + "\n" + str(args[mid_point:])
        return title

    def getLabel(self, label_on, args): 
        label_on_string = ""
        for label in label_on:
            if label in args:
                label_on_string += label+"_"+str(args[label])+"_"
            else:
                label_on_string += label
        return label_on_string

    def destroy(self):
        plt.close()

class PhotometrySinglePlot(PhotometryPlot):
    
    def __init__(self, title=None):
        super().__init__(title)
        self.FS, self.FSX  = plt.subplots()
        self.FS.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        self.outputfile = None

    def addData(self, photometry_csv_file, args, sim_params, label_on, integration, as_baseline=False):
        data = super().getData(photometry_csv_file, integration)
        
        label_on_string = self.getLabel(label_on, args)

        window_size = data["valmax"] - data["valmin"]

        if integration == "t":
            _ = self.FSX.scatter(data["x"], data["y"], s=0.1, label=label_on_string, color=PhotometryPlot.colors[self.ccount])
            _ = self.FSX.errorbar(data["x"], data["y"], xerr=window_size, yerr=data["err"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            if sim_params["onset"] > 0:
                _ = self.FSX.axvline(x=sim_params["onset"], color="red", linestyle="--")

        elif integration == "e":
            _ = self.FSX.bar(data["x"], data["y"], yerr=data["err"], label=label_on_string , color=PhotometryPlot.colors[self.ccount], alpha=0.1, width=args["e_window_step"])
        
        self.ccount += 1

        self.FSX.set_title(self.getTitle(label_on, args))
        self.FSX.set_ylabel('Counts')
        self.FSX.set_xlabel(f'Window center (integration: "{integration}")')
        self.FSX.legend(loc="best")

    def show(self):
        plt.show()


    def save(self, outfileName):
        outfileName = outfileName+'.png'
        self.FS.savefig(outfileName)
        print("Produced: ", outfileName)
        
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

    def addData(self, photometry_csv_file, args, sim_params, label_on, integration, as_baseline=False):
        data = super().getData(photometry_csv_file, integration)

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
                plotargs = {"color": "black"}

            axis.scatter(data["x"], data["y"], label = label_on_string, **plotargs, s=0.1)
            axis.errorbar(data["x"], data["y"], xerr=window_size, yerr=data["err"], fmt="o", **plotargs)

            if not as_baseline:
                if sim_params["onset"] > 0:
                    axis.axvline(x=sim_params["onset"], color="red", linestyle="--")


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