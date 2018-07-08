#!/bin/sh

set -ev

sudo apt-get update

sudo apt-get --yes install libglu1-mesa-dev libgl1-mesa-glx
sudo apt-get --yes install libglew-dev
sudo apt-get --yes install libglfw3-dev
sudo apt-get --yes install libjsoncpp-dev
sudo apt-get --yes install libeigen3-dev
sudo apt-get --yes install libpng-dev
sudo apt-get --yes install libpng16-dev
sudo apt-get --yes install libjpeg-dev
sudo apt-get --yes install python-dev python-tk
sudo apt-get --yes install python3-dev python3-tk
