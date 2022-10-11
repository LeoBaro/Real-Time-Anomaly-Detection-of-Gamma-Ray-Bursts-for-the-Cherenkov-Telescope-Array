import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from RTAscience.aph.photometry import Photometrics
from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.plotting.PlotConfig import PlotConfig
from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils
from rtapipe.lib.datasource.integrationstrat.IntegrationStrategies import IntegrationType


plt.style.use('ieee')

pc = PlotConfig()

def get_max_val(data):
    max_val = 0
    for energy_range in data.keys():
        for value in data[energy_range]:
            if value > max_val:
                max_val = value
    return max_val
#                  T=1  T=5  T=10
# {'(0.04, 0.2)': [0.96, 4.8, 9.6], '(0.2, 1.0)': [0.09, 0.45, 0.9]}
def bar_plot(ax, data, errors, labels, ylabel):

    x = np.arange(len(labels))*3/2  # the label locations
    width = 0.30  # the width of the bars

    rects = []
    count = 0
    for energy_range in data.keys():

        energy_range_data = data[energy_range]

        if errors:
            yerr = errors[energy_range]
        else:
            yerr = None

        shift = x + width/2 - width*len(data)/2 + count*width
        rects.append(
            ax.bar(shift, energy_range_data, width, yerr=yerr, color=pc.colors[count], label=energy_range+" TeV")
        )
        count += 1 

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(f'Number of energy bins = {len(data)}', fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, get_max_val(data)*1.3)

    for rect in rects:
        ax.bar_label(rect, padding=3, fontsize=8)
    
    fig.tight_layout()
    # ax.autoscale(tight=True)
    # ax.set_ylim(0, 15)
    ax.legend(loc='upper left')


# m = (a +/- da) + (b +/- db)  => è somma delle misure
# dm = sqrt(da^2 + db^2) 
# media(dm) = dm / len(m)
def get_error_of_mean(data):
    errors = np.sqrt(data)
    errors = errors ** 2
    errors = np.sqrt(np.sum(errors))
    return errors / len(data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pld', '--photon-lists-dir', type=str, default="/scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/backgrounds", help='DL3 files directory')
    parser.add_argument('-od',  '--output_dir', type=str, default="./tmp-output", help='Output dir')
    parser.add_argument('--tobs', type=int,   default=100, help='The DL3 lenght in seconds')
    parser.add_argument('--emin', type=float, default=0.04)
    parser.add_argument('--emax', type=float, default=1.0)
    args = parser.parse_args()

    # the first photon list inside the photon-lists-dir directory
    input_photon_list = Path(args.photon_lists_dir).joinpath("runid_notemplate_trial_0000010001_simtype_bkg_onset_0_delay_0_offset_0.5.fits")

    integration_times = [1,5,10]
    energy_bins = [1,2,3,4]

    # For the same region of 0.5 degrees radius, 
    # a plot for each integration time will be produced. 
    # The plot will show the average and dev of counts in the region
    # for different energy integration levels.
    ph = Photometry2(Path(__file__).parent.resolve().joinpath("config.yml"), args.photon_lists_dir, args.output_dir)
    photometrics = Photometrics({'events_filename': input_photon_list })

    region = ph.computeRegion(0.2)
    print(f"Region: {region}")

    area_eff_dict = {}
    for eb in energy_bins:
        eWindows = PhotometryUtils.getLogWindows(args.emin, args.emax, eb)
        area_eff = ph.computeAeffArea(eWindows, region)
        area_eff_dict[eb] = area_eff
        print(f"Area eff: {area_eff_dict}")


    energy_bins_data = {}
    energy_bins_data_errors = {}
    energy_bins_fluence_data = {}
    energy_bins_fluence_data_errors = {}
    energy_bins_flux_data = {}
    energy_bins_flux_data_errors = {}

    for eb in energy_bins:
        print(f"\n\nNumber of energy bins: {eb}")

        energy_bins_data[eb] = {}
        energy_bins_data_errors[eb] = {}
        energy_bins_fluence_data[eb] = {}
        energy_bins_fluence_data_errors[eb] = {}
        energy_bins_flux_data[eb] = {}
        energy_bins_flux_data_errors[eb] = {}

        eWindows = PhotometryUtils.getLogWindows(args.emin, args.emax, eb)
        eWindows.reverse()
        print("eWindows: ", eWindows)

        for eWindow in eWindows:
            print(f"eWindow: {eWindow}")

            energy_bins_data[eb][str(eWindow)] = []
            energy_bins_data_errors[eb][str(eWindow)] = []
            energy_bins_fluence_data[eb][str(eWindow)] = []
            energy_bins_fluence_data_errors[eb][str(eWindow)] = []
            energy_bins_flux_data[eb][str(eWindow)] = []
            energy_bins_flux_data_errors[eb][str(eWindow)] = []

            for it in integration_times:
                print("integration time: ", it)
                tWindows = PhotometryUtils.getLinearWindows(0, args.tobs, int(it), int(it))

                counts = []
                fluence_counts = []
                flux_counts = []
                for twindow in tWindows:

                    c = photometrics.region_counter(region, float(region["rad"]), tmin=twindow[0], tmax=twindow[1], emin=eWindow[0], emax=eWindow[1])

                    fluence_c = c / area_eff_dict[eb][str(eWindow)]

                    livetime = twindow[1] - twindow[0]
                    flux_c = fluence_c / livetime

                    counts.append(c)
                    fluence_counts.append(fluence_c)
                    flux_counts.append(flux_c)

                c_mean = round(np.mean(counts),2)
                c_std_dev = round(get_error_of_mean(counts),2)
                print(f"Average counts: {c_mean} +- {c_std_dev}")

                c_fluence_mean = np.mean(fluence_counts)
                c_fluence_std_dev = get_error_of_mean(fluence_counts)
                print(f"Average fluence counts: {c_fluence_mean} +- {c_fluence_std_dev}")

                c_flux_mean = np.mean(flux_counts)
                c_flux_std_dev = get_error_of_mean(flux_counts)
                print(f"Average flux counts: {c_flux_mean} +- {c_flux_std_dev}")

                energy_bins_data[eb][str(eWindow)].append(c_mean)
                energy_bins_data_errors[eb][str(eWindow)].append(c_std_dev)
                energy_bins_fluence_data[eb][str(eWindow)].append(c_fluence_mean)
                energy_bins_fluence_data_errors[eb][str(eWindow)].append(c_fluence_std_dev)
                energy_bins_flux_data[eb][str(eWindow)].append(c_flux_mean)
                energy_bins_flux_data_errors[eb][str(eWindow)].append(c_flux_std_dev)

    print("counts: ",        energy_bins_data)
    print("counts_errors: ", energy_bins_data_errors)
    print("fluence: ",       energy_bins_fluence_data)
    print("fluence_errors: ",energy_bins_fluence_data_errors)
    print("flux: ",          energy_bins_flux_data)
    print("flux_errors: ",   energy_bins_flux_data_errors)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    fig.suptitle('Photon counts by energy range and integration time', fontsize=14)
    axes = axes.ravel()
    bar_plot(axes[0], energy_bins_data[1], energy_bins_data_errors[1], ["1s", "5s", "10s"], ylabel="Counts")
    bar_plot(axes[1], energy_bins_data[2], energy_bins_data_errors[2], ["1s", "5s", "10s"], ylabel="Counts")
    bar_plot(axes[2], energy_bins_data[3], energy_bins_data_errors[3], ["1s", "5s", "10s"], ylabel="Counts")
    bar_plot(axes[3], energy_bins_data[4], energy_bins_data_errors[4], ["1s", "5s", "10s"], ylabel="Counts")
    plt.savefig(f"./counts_by_energy_range_and_integration_time.png", dpi=300)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    fig.suptitle('Fluence(ph/cm2) by energy range and integration time', fontsize=14)
    axes = axes.ravel()
    bar_plot(axes[0], energy_bins_fluence_data[1], None, ["1s", "5s", "10s"], ylabel="Fluence(ph/cm2)")
    bar_plot(axes[1], energy_bins_fluence_data[2], None, ["1s", "5s", "10s"], ylabel="Fluence(ph/cm2)")
    bar_plot(axes[2], energy_bins_fluence_data[3], None, ["1s", "5s", "10s"], ylabel="Fluence(ph/cm2)")
    bar_plot(axes[3], energy_bins_fluence_data[4], None, ["1s", "5s", "10s"], ylabel="Fluence(ph/cm2)")
    plt.savefig(f"./fluence_by_energy_range_and_integration_time.png", dpi=300)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    fig.suptitle('Flux(ph/cm2/s) by energy range and integration time', fontsize=14)
    axes = axes.ravel()
    bar_plot(axes[0], energy_bins_flux_data[1], None, ["1s", "5s", "10s"], ylabel="Flux(ph/cm2/s)")
    bar_plot(axes[1], energy_bins_flux_data[2], None, ["1s", "5s", "10s"], ylabel="Flux(ph/cm2/s)")
    bar_plot(axes[2], energy_bins_flux_data[3], None, ["1s", "5s", "10s"], ylabel="Flux(ph/cm2/s)")
    bar_plot(axes[3], energy_bins_flux_data[4], None, ["1s", "5s", "10s"], ylabel="Flux(ph/cm2/s)")
    plt.savefig(f"./flux_by_energy_range_and_integration_time.png", dpi=300)

