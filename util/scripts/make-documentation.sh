#!/bin/sh

# open3d.org/docs
pip install sphinx sphinx-autobuild sphinx-rtd-theme
cd ../../docs && make html && cd ../util/scripts

# open3d.org/cppapi
sudo apt-get -y install doxygen
cd ../../
doxygen Doxyfile
