#!/usr/bin/python
# -*- coding: latin-1 -*-
from setuptools import setup, find_packages
setup( name='rtapipe',
       version='0.0.0',
       author='Leonardo Baroncelli',
       author_email='leonardo.baroncelli@inaf.it',
       packages=find_packages(),
       package_dir={ 'rtapipe': 'rtapipe'},
       include_package_data=True
     )