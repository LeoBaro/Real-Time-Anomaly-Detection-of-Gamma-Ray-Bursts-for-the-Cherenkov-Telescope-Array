#!/usr/bin/python

from setuptools import setup, find_packages

entry_points = {
	'console_scripts': [
		'plot_ap_timeseries = rtapipe.scripts.plots.plot_ap_timeseries:main',
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