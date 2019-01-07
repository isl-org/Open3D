#!/bin/bash

# list of the supported ubuntu versions
declare -a ubuntu_version=(14.04 16.04 18.04)

# base builds don't install Open3D 3rdparty dependencies
# deps builds install the Open3D 3rdparty dependencies
declare -a bundle_type=(base deps)

# py2/3 represent the native environment with python2/3
# mc2/3 represent the miniconda2/3 environment
declare -a env_type=(py2 py3 mc2 mc3)

# type of linking to be used at build time
declare -a link_type=(STATIC SHARED)
