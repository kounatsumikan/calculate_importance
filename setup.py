#!/usr/bin/env python

from distutils.core import setup

NAME='calculate_importance'



setup(name=NAME,
      version='1.2',
      description='calculate',
      author='Kou Natsuhara',
      packages=[NAME, "utils"],
      package_dir={NAME: '', "utils": "utils"},
      install_requires=["numpy", "pandas", "sklearn", "minepy"]
)