# MAINTAINER Dan Petre dan.petre@intel.com

FROM ubuntu:16.04
# FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

ENV http_proxy  "http://proxy-chain.intel.com:911"
ENV HTTP_PROXY  "http://proxy-chain.intel.com:911"
ENV https_proxy "http://proxy-chain.intel.com:912"
ENV HTTPS_PROXY "http://proxy-chain.intel.com:912"
ENV ftp_proxy   "http://proxy-chain.intel.com:911"
ENV socks_proxy "http://proxy-us.intel.com:1080"
ENV no_proxy    "intel.com;.intel.com;localhost;127.0.0.1"
ENV ALL_PROXY   "socks5://proxy-us.intel.com"

ENV DEBIAN_FRONTEND noninteractive

# upgrade existing
RUN apt-get update
RUN apt-get upgrade -y

# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y --no-install-recommends apt-utils

# Install xvfb
RUN apt-get install -y xvfb

# Install vnc
RUN apt-get install -y x11vnc

# simple window manager
RUN apt-get install -y ratpoison

# standard terminal emulator for the X Window System
RUN apt-get install -y xterm

# # standard lxde terminal eumulator without dependencies
# RUN apt-get install -y lxterminal

# useful scripts for adding and removing repositories (like PPAs)
RUN apt-get install -y software-properties-common

RUN apt-get install -y locales
RUN apt-get --reinstall install xfonts-base

RUN apt-get install -y git

# Open3D dependencies
RUN apt-get install -y xorg-dev
RUN apt-get install -y libglu1-mesa-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglew-dev
RUN apt-get install -y libglfw3-dev
RUN apt-get install -y libjsoncpp-dev
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y libjpeg-dev
RUN apt-get install -y python-dev
RUN apt-get install -y python3-dev
RUN apt-get install -y cmake

# upgrade existing
RUN apt-get update
RUN apt-get upgrade -y
# RUN apt-get autoremove

# set language
RUN locale-gen en_US.UTF-8
RUN update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN mkdir ~/.vnc
# Setup a password
RUN x11vnc -storepasswd 1234 ~/.vnc/passwd

# start virtual framebuffer and ratpoison
RUN Xvfb -screen 0 800x600x32 -ac &
RUN DISPLAY=:0 ratpoison &

EXPOSE 5920

RUN cd ~ && git clone https://github.com/IntelVCL/Open3D

# DONE above
# RUN cd ~/Open3D && ./scripts/install-deps-ubuntu.sh
RUN cd ~/Open3D && mkdir build
RUN cd ~/Open3D/build && cmake ../src
RUN cd ~/Open3D/build && make -j

CMD ["x11vnc", "-forever", "-usepw", "-create"]
