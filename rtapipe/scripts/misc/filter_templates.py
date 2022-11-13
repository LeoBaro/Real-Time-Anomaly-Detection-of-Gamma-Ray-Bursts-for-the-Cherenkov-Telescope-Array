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

def main():
    parser = argparse.ArgumentParser(description='Simulate empty fields.')
    parser.add_argument('-cp', '--catalog-path', type=str, default='/scratch/baroncelli/DATA/templates/grb_afterglow/GammaCatalogV1.0', help='absolute path where cat is installed')
    parser.add_argument('-g', '--generate', type=bool, default=False, help='')
    args = parser.parse_args()
    
    if args.generate:
        generate_data(args.catalog_path)
    else:
        templates = load_data()

    templates = [t for t in templates if "ebl" not in t.name]

    filtered_templates = filter_templates(templates, 1e-9)
    # write template names on file
    with open("filtered_templates.txt", "w") as f:
        f.write("[")
        for template in filtered_templates:
            f.write(template.name.replace(".fits", "")+",")
        f.write("]")

    plot_max_flux_distribution(templates, "all templates", f"GammaCatalogV1.0 ({len(templates)} templates)")
    plot_max_flux_distribution(filtered_templates, "filtered", "Flux > 1e-11")

def plot_max_flux_distribution(templates, name, title=""):
    max_fluxes = []
    for template in templates:
        max_fluxes.append(np.max(template.flux))
    max_fluxes = np.array(max_fluxes)
    
    #max_fluxes = (max_fluxes-max_fluxes.min())/(max_fluxes.max()-max_fluxes.min())
    #print(max_fluxes.max(), max_fluxes.min())
    #print(max_fluxes[0:100])
    #max_fluxes = max_fluxes[max_fluxes<0.1] 
    #print("Number of filtered templates < 0.1: ", len(max_fluxes))

    print("Producing histogram...")
    pc = PlotConfig()
    fig, ax = plt.subplots(1,1,figsize=pc.fig_size)
    fig.suptitle(f"Distribution of the maximum flux of the templates", fontsize=pc.fig_suptitle_size)
    ax.set_title(title, fontsize=pc.fig_title_size)
    _ = ax.hist(max_fluxes, bins=np.logspace(np.log10(1e-11),np.log10(1e-4), 100), **pc.get_histogram_colors())
    # vertical line
    ax.axvline(x=3.3e-09, color='grey', linestyle='--', label="run0406_ID000126", linewidth=1)
    ax.axvline(x=1e-09, color='black', linestyle='--', label="Background for North_z40_5h_LST", linewidth=1.5)

    ax.set_xlabel("Max flux (log)")
    ax.set_ylabel("Counts (log)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f"max_flux_distribution_{name}.png", dpi=pc.dpi) 
    plt.close()



def filter_templates(templates, threshold):
    filtered_templates = []
    for template in templates:
        if template.name == "run0406_ID000126.fits":
            print("Max flux of run0406_ID000126: ", np.max(template.flux))
        
        if np.max(template.flux) > threshold:
            filtered_templates.append(template)
    print("Number of filtered templates: ", len(filtered_templates))
    return filtered_templates


if __name__=='__main__':
    main()
