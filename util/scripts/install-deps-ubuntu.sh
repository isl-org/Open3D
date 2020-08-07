#!/usr/bin/env bash
set -ev

sudo apt-get update

if [ "$1" == "assume-yes" ]; then
    # Open3D deps
    sudo apt-get --yes install xorg-dev
    sudo apt-get --yes install libglu1-mesa-dev
    sudo apt-get --yes install python3-dev
    # Filament build-from-source dpes
    sudo apt-get --yes install libsdl2-dev
    sudo apt-get --yes install libc++-7-dev
    sudo apt-get --yes install libc++abi-7-dev
    sudo apt-get --yes install ninja-build
    sudo apt-get --yes install libxi-dev
    # ML deps
    sudo apt-get --yes install libtbb-dev
    # Headless rendering deps
    sudo apt-get --yes install libosmesa6-dev
else
    # Open3D deps
    sudo apt-get install xorg-dev
    sudo apt-get install libglu1-mesa-dev
    sudo apt-get install python3-dev
    # Filament build-from-source dpes
    sudo apt-get install libsdl2-dev
    sudo apt-get install libc++-7-dev
    sudo apt-get install libc++abi-7-dev
    sudo apt-get install ninja-build
    sudo apt-get install libxi-dev
    # ML deps
    sudo apt-get install libtbb-dev
    # Headless rendering deps
    sudo apt-get install libosmesa6-dev
fi
