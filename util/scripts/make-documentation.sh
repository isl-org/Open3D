#!/usr/bin/env bash
# Documentation build scripts for CI
#
# To build documentation locally, ignore the xvfb-run and arguments.
#
# Prerequisites:
# pip install sphinx sphinx-autobuild
# sudo apt-get -y install doxygen

set -e
curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

pushd ${curr_dir}/../../docs
python make_docs.py --clean_notebooks --execute_notebooks=auto --sphinx --doxyge
popd
