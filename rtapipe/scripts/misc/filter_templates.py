import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from RTAscience.lib.RTAVisualise import get_template_lc
from rtapipe.lib.plotting.PlotConfig import PlotConfig
@dataclass
class Template:
    name: str
    time: np.ndarray
    flux: np.ndarray

def generate_data(catalog_path):
    templates = []
    for file in tqdm(os.listdir(catalog_path)):
        if file.endswith('.fits'):
            time, flux = get_template_lc(runid=file, erange=[0.04, 1], path=catalog_path)
            templates.append(Template(name=file, time=time, flux=flux))
    np.save('templates.npy', templates)

def load_data():
    templates = np.load('templates.npy', allow_pickle=True)
    return templates




def filter_templates(templates, threshold_min, threshold_max):
    selected_templates = []
    not_selected_templates = []
    max_flux = np.max(templates[0].flux)
    
    print("Filtering with threshold_min: ", threshold_min, " and threshold_max: ", threshold_max, "")
    
    for template in templates:


        if np.max(template.flux) >= threshold_min and np.max(template.flux) <= threshold_max:
            selected_templates.append(template)
            if np.max(template.flux) > max_flux:
                max_flux = np.max(template.flux)

        else:
            not_selected_templates.append(template)

    return selected_templates, not_selected_templates, max_flux

def get_max_fluxes(templates):
    max_fluxes = []
    for template in templates:
        max_fluxes.append(np.max(template.flux))
    return np.array(max_fluxes)

def write_templates(templates, filename):
    with open(f"{filename}.txt", "w") as f:
        f.write("[")
        for i,template in enumerate(templates):
            f.write(template.name.replace(".fits", ""))
            if i == len(templates)-1:
                f.write("]")
            else:
                f.write(",")
 

def main():
    """
    Background level: 
        2.678473678188729e-10 < 1.226381962327406e-09 (irf) < inf
    """


    parser = argparse.ArgumentParser(description='Simulate empty fields.')
    parser.add_argument('-cp', '--catalog-path', type=str, default='/scratch/baroncelli/DATA/templates/grb_afterglow/GammaCatalogV1.0', help='absolute path where cat is installed')
    parser.add_argument('-g', '--generate', type=bool, default=False, help='')
    parser.add_argument('-p', '--plot', type=int, default=0, choices=[0,1], help='')
    parser.add_argument('-t', '--template', type=str, default=False, help='')

    args = parser.parse_args()
    
    if args.generate:
        generate_data(args.catalog_path)
    else:
        templates = load_data()

    templates = [t for t in templates if "ebl" not in t.name]

    if args.plot == 1:
        fig, ax = plt.subplots(1,1,figsize=(10,10))
        for t in templates:
            if args.template+".fits" == t.name:
                print(t.name)
                ax.plot(t.time, t.flux)
                ax.set_xlim(0, 800)
                fig.savefig(f"template_{t.name}.png")
                

    irf_bg = 1.226381962327406e-09
    sigma_1_neg = 2.678473678188729e-10
    print("Total number of templates: ", len(templates))

    test_set_H_templates, other_templates, max_flux_h = filter_templates(templates, sigma_1_neg, irf_bg)
    print("Number of templates in test set H: ", len(test_set_H_templates))
    print("Number of templates in other set: ", len(other_templates))
    print("Max flux h : ", max_flux_h)

    test_set_A_templates, other_templates, max_flux_e = filter_templates(templates, sigma_1_neg, 1)
    print("Number of templates in test set A: ", len(test_set_A_templates))
    print("Number of templates in other set: ", len(other_templates))
    print("Max flux e: ", '{:.4E}'.format(max_flux_e))


    


    #flux_range_E = f"{irf_bg:.3E} <= Flux <= {max_flux_e:.3E}"
    #flux_range_H = f"{sigma_1_neg:.3E} <= Flux <= {max_flux_h:.3E}"

    write_templates(test_set_A_templates, f"selected_templates_test_set_A")
    write_templates(test_set_H_templates, f"selected_templates_test_set_H")

    plot_max_flux_distribution(get_max_fluxes(test_set_H_templates), get_max_fluxes(test_set_A_templates), get_max_fluxes(other_templates), irf_bg, sigma_1_neg)


def plot_max_flux_distribution(test_set_H_templates, test_set_A_templates, other_templates, irf_bg, sigma_1_neg):

    print("Producing histogram...")
    pc = PlotConfig()
    fig, ax = plt.subplots(1,1,figsize=pc.fig_size)
    fig.suptitle(f"Distribution of the peak flux of the templates", fontsize=pc.fig_suptitle_size)
     
    _ = ax.hist(other_templates, label="Not selected", 
                    bins=np.logspace(np.log10(1e-11),np.log10(1e-4), 125), 
                    lw=0.8, ls='dashed', color="white", ec="black", alpha=0.8)
    """
    _ = ax.hist(test_set_H_templates, label="Test Set H",
                    bins=np.logspace(np.log10(1e-11),np.log10(1e-4), 125), 
                    lw=1.5, color="black", ec="black", alpha=0.3)
    """
    _ = ax.hist(test_set_A_templates, label="Test Set Templates",
                    bins=np.logspace(np.log10(1e-11),np.log10(1e-4), 125), 
                    lw=1.5, color="black", ec="black", alpha=0.3)

    # vertical line
    #ax.axvline(x=3.3e-09, color='grey', linestyle='--', label="run0406_ID000126", linewidth=1)
    ax.axvline(x=irf_bg, color='black', linestyle='--', linewidth=1)
    plt.text(irf_bg, 30, f"  Background level mean = {irf_bg:.3E}", fontsize=15)

    ax.axvline(x=sigma_1_neg, color='black', linestyle='--', linewidth=1)
    plt.text(sigma_1_neg, 70, f"<--- 1σ --->", fontsize=15)

    ax.set_xlabel(r"Peak flux $erg/s/cm^2$ (log)")
    ax.set_ylabel("Counts (log)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"templates_max_flux_distributions.png", dpi=pc.dpi) 
    plt.close()


if __name__=='__main__':
    main()
