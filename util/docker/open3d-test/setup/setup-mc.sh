#!/bin/bash

# display help on the required command line arguments
if [ $# -eq 0 ] || [ $# -eq 1 ] || [ "${1}" = "--help" ]; then
    echo "./setyp-py.sh <miniconda_installer_file_name> <install_directory>"
    echo
    echo "Required:"
    echo "    Miniconda installer file name"
    echo "    Install directory"
    echo
    exit 1
fi

MC_INSTALLER=${1}
CONDA_DIR=${2}

# install miniconda
/bin/bash "/root/${MC_INSTALLER}" -bfp "${CONDA_DIR}" >/dev/null 2>&1

# install conda-build & conda-verify
"${CONDA_DIR}/bin/conda" install -qy conda-build conda-verify >/dev/null 2>&1

# conda update
"${CONDA_DIR}/bin/conda" update --all -qy >/dev/null 2>&1
"${CONDA_DIR}/bin/conda" config --set auto_update_conda False

# enable the 'conda' command
ln -s "${CONDA_DIR}/etc/profile.d/conda.sh" /etc/profile.d/conda.sh
echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc

# activate when bash starts
echo "conda activate" >> ~/.bashrc

# cleanup
rm -rf "/root/${MC_INSTALLER}"
"${CONDA_DIR}/bin/conda" clean -aqy
