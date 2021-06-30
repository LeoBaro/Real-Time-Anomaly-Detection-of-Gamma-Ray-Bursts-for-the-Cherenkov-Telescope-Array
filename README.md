# phd

## Installation
After cloning the repository:
```bash
git submodule init
git submodule update
```

Since the ctools software is required by the cta-sag-sci submodule, it is convenient to create
an Anaconda virtual environment and install it from the CTA anaconda channel.
```bash
conda env create --name <envname> --file=cta-sag-sci/environment.yaml
```

## Issues

* phlists_to_photometry_plot.py needs X connection, otherwise it raises: _tkinter.TclError: couldn't connect to display "localhost:10.0"

