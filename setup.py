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


def get_gitvers(version=VERSION):

    try:
        process = subprocess.Popen(
            ['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE)
        git_head_hash = process.communicate()[0].strip()
        git_head_hash = git_head_hash.decode("utf-8")
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

    license="LICENSE",
    description="Image Forward Modeling",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"],
                  "forcepho": ["src/*.cu", "src/*.cc", "src/*.h", "src/*.hh"]},
    #scripts=glob.glob("scripts/*.py"),
    include_package_data=True,
    install_requires=["numpy"],
)
