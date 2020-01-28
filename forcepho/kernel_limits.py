#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""kernel_limits.py

We store some compile-time constants for the CUDA kernel in kernel_limits.h.
These set the sizes of the structs we use to communicate with the GPU. Thus, we
need to know them in Python as well.  This class parses the kernel_limits.h
file to extract the necessary info.
"""

import re
import os.path

fn = 'kernel_limits.h'

# TODO: Make this package resources or another way to avoid using __file__?
with open(os.path.join(os.path.dirname(__file__), fn), 'r') as fp:
    lines = fp.readlines()

regexp = re.compile(r'^\s*#define\s+(?P<name>\w+)\s+(?P<value>.+)')

macros = {}
for line in lines:
    match = regexp.match(line)
    if match:
        name, value = match.group('name'), match.group('value')

        # All values should either be int or float (probably)
        try:
            value = int(value)
        except ValueError:
            value = float(value)

        macros[name] = value

# better way to do this?
locals().update(macros)

__all__ = list(macros)
