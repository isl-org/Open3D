#!/usr/bin/env bash
set -e

curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# prerequisites
# pip install sphinx sphinx-autobuild sphinx-rtd-theme
# sudo apt-get -y install doxygen

cd ${curr_dir}/../../docs

# open3d.org/docs
make html

# open3d.org/cppapi
doxygen Doxyfile
