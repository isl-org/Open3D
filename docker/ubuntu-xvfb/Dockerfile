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

# Lightweight X11 Desktop Environment
# RUN apt-get install -y lxde-core

# simple window manager
# RUN apt-get install -y ratpoison

# standard terminal emulator for the X Window System
RUN apt-get install -y xterm

# # standard lxde terminal eumulator without dependencies
# RUN apt-get install -y lxterminal

# useful scripts for adding and removing repositories (like PPAs)
RUN apt-get install -y software-properties-common

RUN apt-get install -y locales
RUN apt-get --reinstall install xfonts-base

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
# RUN DISPLAY=:0 ratpoison &

EXPOSE 5920
CMD ["x11vnc", "-forever", "-usepw", "-create"]
