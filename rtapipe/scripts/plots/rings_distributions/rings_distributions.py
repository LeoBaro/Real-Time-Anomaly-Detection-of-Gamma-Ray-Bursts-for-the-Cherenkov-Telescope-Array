from pathlib import Path
import matplotlib.pyplot as plt

from rtapipe.lib.datasource.Photometry3 import SimulationParams, OnlinePhotometry

def plot_distribution(data, data_err, title, xlabel, ylabel, labels, filename):
    print(f"Plotting {title}")
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    
    x = np.arange(len(labels))*3/2  # the label locations
    width = 0.30  # the width of the bars

    rects = []
    count = 0

    for energy_range in range(data.shape[1]):
        shift = x + width/2 - width*len(data)/2 + count*width
        rects.append(
            ax.bar(shift, data[:,energy_range].sum(), width, label=labels[energy_range])
        )
        count += 1     

    for rect in rects:
        ax.bar_label(rect, padding=3, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.savefig(filename)




if __name__=='__main__':

    sim_params = SimulationParams(runid="run0406_ID000126", simtype="bkg", onset=250, emin=0.04, emax=1, tmin=0, tobs=18000, offset=0.5, irf="North_z40_5h_LST", roi=2.5, caldb="prod5-v0.1")
    test_fits = Path(__file__).parent.joinpath("fits_data")
    output_dir = Path(__file__).parent.joinpath("output")
    add_target_region = False
    remove_overlapping_regions_with_target = False
    compute_effective_area_for_normalization = True
    normalize = True
    with_metadata = True
    integrate_from_regions = "bkg"

    """
    rings_indexes = None # [0,1,2,3,]
    online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)        
    online_photometry.preconfigure_regions(regions_radius=0.2, max_offset=2.0, example_fits=test_fits, add_target_region=add_target_region, remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, compute_effective_area_for_normalization=compute_effective_area_for_normalization, rings_indexes=rings_indexes)
    counts_data, counts_data_err, flux_data, flux_data_err, metadata = online_photometry.integrate(test_fits, normalize, 10, with_metadata, integrate_from_regions=integrate_from_regions)
    for ring, values in online_photometry.regions_config.rings.items():
        print("Ring offset", ring, " with regions ", len(values))
    print(counts_data.shape)
    """
    import numpy as np
    from tqdm import tqdm
    for rings_indexes in [ [0,1,2,3,4] ]:
        
        online_photometry = OnlinePhotometry(sim_params, integration_time=5, tsl=5,  number_of_energy_bins=3)
        online_photometry.preconfigure_regions(
            regions_radius=0.2, 
            max_offset=2.0, 
            example_fits=Path(__file__).parent.joinpath("fits_data", "runid_run0406_ID000126_trial_0000000002_simtype_bkg_onset_0_delay_0_offset_0.5_tobs_18000.fits"), 
            add_target_region=add_target_region, 
            remove_overlapping_regions_with_target=remove_overlapping_regions_with_target, 
            compute_effective_area_for_normalization=compute_effective_area_for_normalization, 
            rings_indexes=rings_indexes
        )
        
        
        distribution_data = []
        distribution_data_err = []
        distribution_data_norm = []
        distribution_data_norm_err = []

        count = 0
        for fits_data in tqdm(test_fits.iterdir()):

            counts_data, counts_data_err, flux_data, flux_data_err, metadata = online_photometry.integrate(fits_data, normalize, 10, with_metadata, integrate_from_regions=integrate_from_regions)
            new_shape = (counts_data.shape[0]*counts_data.shape[1],counts_data.shape[2])        
            distribution_data.append(counts_data.reshape(new_shape))
            distribution_data_err.append(counts_data_err.reshape(new_shape))
            distribution_data_norm.append(flux_data.reshape(new_shape))
            distribution_data_norm_err.append(flux_data_err.reshape(new_shape))
            count += 1
            if count == 5:
                break
            

        distribution_data = np.array(distribution_data)
        distribution_data_err = np.array(distribution_data_err)
        distribution_data_norm = np.array(distribution_data_norm)
        distribution_data_norm_err = np.array(distribution_data_norm_err)
        new_shape = (distribution_data.shape[0]*distribution_data.shape[1],distribution_data.shape[2])        
        
        distribution_data = distribution_data.reshape(new_shape)
        distribution_data_err = distribution_data_err.reshape(new_shape)
        distribution_data_norm = distribution_data_norm.reshape(new_shape)
        distribution_data_norm_err = distribution_data_norm_err.reshape(new_shape)

        print("Background mean level:")
        print(distribution_data.sum(axis=1).mean(axis=0),"+-",distribution_data.sum(axis=1).std(axis=0))
        print(distribution_data_norm.sum(axis=1).mean(axis=0),"+-",distribution_data_norm.sum(axis=1).std(axis=0))

        limit = 125
        distribution_data = distribution_data[0:limit, :]
        distribution_data_err = distribution_data_err[0:limit, :]
        distribution_data_norm = distribution_data_norm[0:limit, :]
        distribution_data_norm_err = distribution_data_norm_err[0:limit, :]
         
        plot_distribution(distribution_data, distribution_data_err,
            "Counts distribution for ring "+str(rings_indexes), 
            "Energy range", 
            "Counts", 
            ["EB_0.04-0.117","EB_2-0.117-0.342","EB_0.342-1"],
            output_dir.joinpath(f"counts_distribution_rings_{rings_indexes[0]}.png")
        )
        plot_distribution(distribution_data_norm, distribution_data_norm_err,
            "Flux distribution for ring "+str(rings_indexes), 
            "Energy range", 
            "Flux", 
            ["EB_0.04-0.117","EB_2-0.117-0.342","EB_0.342-1"],
            output_dir.joinpath(f"flux_distribution_rings_{rings_indexes[0]}.png")
        )        