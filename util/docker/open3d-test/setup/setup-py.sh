#!/bin/bash

# options for the 1st argument
declare -a python_version=(python python3)

# display help on the required command line arguments
if [ $# -eq 0 ] || [ "${1}" = "--help" ]; then
    echo "./setyp-py.sh <python_version>"
    echo
    echo "Required:"
    echo "    Python version:   ${python_version[*]}"
    echo
    exit 1
fi

# display help on the first required argument
if [[ ! " ${python_version[@]} " =~ " ${1} " ]]; then
    echo "    options for the the 1st argument: ${python_version[*]}"
    echo "    argument provided: '${1}'"
    echo
    exit 1
fi

PYTHON=${1}

# install native python dependencies
apt-get update -qq
apt-get install -qq -y --no-install-recommends \
    ${PYTHON}-dev \
    ${PYTHON}-pip \
    ${PYTHON}-setuptools \
    ${PYTHON}-wheel >/dev/null 2>&1

# cleanup
rm -rf /var/lib/apt/lists/*
