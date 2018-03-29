# MAINTAINER Dan Petre dan.petre@intel.com

FROM ubuntu:16.04

ENV http_proxy "http://proxy-chain.intel.com:911"
ENV HTTP_PROXY "http://proxy-chain.intel.com:911"
ENV https_proxy "http://proxy-chain.intel.com:912"
ENV HTTPS_PROXY "http://proxy-chain.intel.com:912"
ENV ftp_proxy "http://proxy-chain.intel.com:911"
ENV socks_proxy "http://proxy-us.intel.com:1080"
ENV no_proxy "intel.com;.intel.com;localhost;127.0.0.1"
ENV ALL_PROXY "socks5://proxy-us.intel.com"

ENV DEBIAN_FRONTEND noninteractive

# install Open3D dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    apt-utils \
    build-essential \
    cmake \
    git \
    gdb \
    # libeigen3-dev \
    # libgl1-mesa-dev \
    # libgl1-mesa-glx \
    # libglew-dev \
    # libglfw3-dev \
    libglu1-mesa-dev \
    # libjpeg-dev \
    # libjsoncpp-dev \
    libosmesa6-dev \
    # libpng-dev \
    lxde \
    # lxterminal \
    ne \
    # python-dev \
    python3-dev \
    python3-pip \
    python3-tk \
    python3.5-dbg \
    software-properties-common \
    x11vnc \
    xorg-dev \
    xterm \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip install \
    matplotlib \
    numpy \
    opencv-python

EXPOSE 5920

ENV HOME=/root

WORKDIR $HOME
COPY ./setup/.xinitrc .
RUN chmod a+x .xinitrc
COPY ./setup/entrypoint.sh .

WORKDIR $HOME/Open3D/build/lib/Tutorial/Advanced

CMD ["bash", "/root/entrypoint.sh"]
