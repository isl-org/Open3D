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
    # lxterminal \
    ne \
    python-dev \
    python3-dev \
    python3-pip \
    python3-tk \
    ratpoison \
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

ENV HOME=/root
WORKDIR $HOME
RUN git clone https://github.com/IntelVCL/Open3D.git

RUN mkdir $HOME/Open3D/build
WORKDIR $HOME/Open3D/build
RUN cmake \
    -DOpen3D_USE_NATIVE_DEPENDENCY_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    ../src
RUN make -j

# debug tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    gdb \
    python3.5-dbg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $HOME
COPY .xinitrc .
RUN chmod a+x .xinitrc
COPY entrypoint.sh .

WORKDIR $HOME/Open3D/build/lib/Tutorial/Advanced
COPY test.py .
RUN chmod a+x test.py

EXPOSE 5920

CMD ["bash", "/root/entrypoint.sh"]
