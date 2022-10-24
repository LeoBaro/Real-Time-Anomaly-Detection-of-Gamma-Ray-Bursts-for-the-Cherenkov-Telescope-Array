#!/usr/bin/python

from setuptools import setup, find_packages

entry_points = {
	'console_scripts': [
		'plot_ap_timeseries = rtapipe.scripts.plots.plot_ap_timeseries:main',
          'generate_ap_data = rtapipe.lib.dataset.generate_ap_data:main',
          'phlists_to_skymaps = rtapipe.scripts.plots.phlists_to_skymaps:main',
          'get_dataset_config_path = rtapipe.lib.dataset.config.get_dataset_config_path:main'
     ]
}
setup( 
     name='rtapipe',
     version='0.0.0',
     author='Leonardo Baroncelli',
     author_email='leonardo.baroncelli@inaf.it',
     packages=find_packages(),
     package_dir={ 'rtapipe': 'rtapipe'},
     include_package_data=True,
	entry_points=entry_points
)