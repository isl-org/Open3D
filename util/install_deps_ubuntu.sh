#!/usr/bin/env bash
set -ev

SUDO=${SUDO:=sudo}      # SUDO=command in docker (running as root, sudo not available)

$SUDO apt-get update

if [ "$1" == "assume-yes" ]; then
    # Open3D deps
    $SUDO apt-get --yes install xorg-dev
    $SUDO apt-get --yes install libglu1-mesa-dev
    $SUDO apt-get --yes install python3-dev
    # Filament build-from-source dpes
    $SUDO apt-get --yes install libsdl2-dev
    $SUDO apt-get --yes install libc++-7-dev
    $SUDO apt-get --yes install libc++abi-7-dev
    $SUDO apt-get --yes install ninja-build
    $SUDO apt-get --yes install libxi-dev
    # ML deps
    $SUDO apt-get --yes install libtbb-dev
    # Headless rendering deps
    $SUDO apt-get --yes install libosmesa6-dev
else
    # Open3D deps
    $SUDO apt-get install xorg-dev
    $SUDO apt-get install libglu1-mesa-dev
    $SUDO apt-get install python3-dev
    # Filament build-from-source dpes
    $SUDO apt-get install libsdl2-dev
    $SUDO apt-get install libc++-7-dev
    $SUDO apt-get install libc++abi-7-dev
    $SUDO apt-get install ninja-build
    $SUDO apt-get install libxi-dev
    # ML deps
    $SUDO apt-get install libtbb-dev
    # Headless rendering deps
    $SUDO apt-get install libosmesa6-dev
fi
