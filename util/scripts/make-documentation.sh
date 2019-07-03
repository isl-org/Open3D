#!/usr/bin/env bash
# Documentation build scripts for CI
#
# Prerequisites:
# pip install sphinx sphinx-autobuild
# sudo apt-get -y install doxygen

set -e
curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

pushd ${curr_dir}/../../docs
python make_docs.py --sphinx --doxyge
popd
