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


VERSION = "0.5"

# from pybind11 import get_cmake_dir
# from pybind11.setup_helpers import Pybind11Extension, build_ext

# srcs =  glob.glob("forcepho/src/*cc") #+ glob.glob("forcepho/src/*h")
# srcs.sort()

# ext_modules = [
#     Pybind11Extension("compute_gaussians_kernel",
#                      srcs,
#                       # Example: passing in the version to the compiled code
#                       # define_macros = [('VERSION_INFO', get_gitvers())],
#                      ),
# ]


def get_gitvers(version=VERSION):

    try:
        process = subprocess.Popen(
            ['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE,
            universal_newlines=True, encoding="utf-8")
        git_head_hash = process.communicate()[0].strip()
        #git_head_hash = git_head_hash.decode("utf-8")
        version = f"{version}+{git_head_hash}"

    except:
        pass

    with open("./forcepho/_version.py", "w") as f:
        f.write(f"__version__ = '{version}'")

    return version


setup(
    name="forcepho",
    url="https://github.com/bd-j/forcepho",
    version=get_gitvers(),
    author="Ben Johnson",
    author_email="benjamin.johnson@cfa.harvard.edu",
    packages=["forcepho",
              "forcepho.patches",
              "forcepho.mixtures",
              "forcepho.slow"],
    ext_modules=ext_modules,
    #license="LICENSE",
    description="Image Forward Modeling",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"],
                  "forcepho": ["src/*.cu", "src/*.cc", "src/*.h", "src/*.hh"]},
    #scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy"],
)
