#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import glob
import subprocess
try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

    
setup(
    name="forcepho",
    url="https://github.com/bd-j/forcepho",
    version="0.2",
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["forcepho",
              "forcepho.boneyard",
              "forcepho.mixtures"],

    license="LICENSE",
    description="Image Forward Modeling",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy"],
)
