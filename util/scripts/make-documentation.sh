#!/bin/sh

# prerequisites
# pip install sphinx sphinx-autobuild sphinx-rtd-theme
# sudo apt-get -y install doxygen

cd ../../docs

# open3d.org/docs
make html

# open3d.org/cppapi
doxygen Doxyfile
