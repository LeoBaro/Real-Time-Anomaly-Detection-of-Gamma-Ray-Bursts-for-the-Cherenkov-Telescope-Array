# phd

## Installation on Intel

Clone with "--recurse-submodule" option.

Anaconda is suggested to create a python virtual environment.
```bash
conda create --name bphb python=3.7
```

Install the python dependencies:
```bash
conda env update --name bphd --file=cta-sag-sci/environment.yaml
conda env update --name bphd --file=environment.yaml
```

Install the packages:
```bash
conda activate bphd
cd cta-sag-sci && python setup.py develop && cd ..
cd astro && python setup.py develop && cd ..
python setup.py develop
```




## Installation of Power9

Clone with "--recurse-submodule" and "--branch=power9" options.

The healpy package is deleted in the branch power9 of the cta-sag-sci submodule because
it will download a lot of dependecies that cause conflicts with astropy+tensorflow-gpu.

Create a python virtual environment with anaconda.

The "ctools", "pyregion" and "xml" anaconda packages are not available on POWER.

Install them manually
* ctools: http://cta.irap.omp.eu/ctools/admin/install_source.html .
* pyregion: install it with pip
* xml: --

Then, you can follow the instruction "Install the python dependencies" and "Install the packages".



## Issues
* phlists_to_photometry_plot.py needs X connection, otherwise it raises: _tkinter.TclError: couldn't connect to display "localhost:10.0"
