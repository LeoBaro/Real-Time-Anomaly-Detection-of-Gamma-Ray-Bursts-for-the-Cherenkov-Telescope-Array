import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 10, 'lines.markersize': 0.5,'legend.markerscale': 5, 'lines.linewidth':0.5, 'lines.linestyle':'--'})

class PhotometryPlot:
    
    inch_x = 10
    inch_y = 6
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:pink","tab:olive","tab:cyan"]
    
    def __init__(self, title=None):
        self.outPlot = None
        self.title = title
        self.ccount = 0

    def getData(self, photometry_csv_file):
        df = pd.read_csv(str(photometry_csv_file))

        try:
            df["TCENTER"] = (df["TMAX"] + df['TMIN'])/2
        except:
            df["TCENTER"] = None

        try:
            df["ECENTER"] = (df["EMAX"] + df['EMIN'])/2
        except:
            df["ECENTER"] = None

        return df

    def getColor(self):
        col = PhotometryPlot.colors[self.ccount]
        self.ccount += 1
        return col

    def getForbidden(self):
        return ["datapath", "simtype", "runid", "input_file", "override", "plot"]

    def addData(self, photometry_csv_file, args, label_on, integration, vertical_line = False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
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
 
    def destroy(self):
        plt.close()

