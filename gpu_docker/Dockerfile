# nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
# nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04
ARG BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
FROM ${BASE_IMAGE}

# Regular deps
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    autoconf \
    ca-certificates \
    git \
    gnupg \
    libc++-7-dev \
    libc++abi-7-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libsdl2-dev \
    libtbb-dev \
    libtool \
    libudev-dev \
    libxi-dev \
    ninja-build \
    python3-dev \
    software-properties-common \
    wget \
    xorg-dev \
 && rm -rf /var/lib/apt/lists/*

# CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
 && apt-add-repository --yes 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt-get install -y \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# CCache
RUN git clone https://github.com/ccache/ccache.git \
 && cd ccache \
 && git checkout v4.2 -b 4.2 \
 && mkdir build \
 && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DZSTD_FROM_INTERNET=ON .. \
 && make -j$(nproc) \
 && make install \
 && ccache --version \
 && ccache -M 10G

# Virtualenv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
 && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/local/bin/python \
 && ln -s /usr/bin/python3-config /usr/local/bin/python-config \
 && ln -s /usr/bin/pip3 /usr/local/bin/pip

# Install ML deps
# No virtual environment is needed, default to python 3.6
RUN python -m pip install -U pip=="20.2.4" wheel=="0.35.1" setuptools=="50.3.2" yapf=="0.30.0"
RUN python -m pip install -U pytest=="6.0.1" scipy=="1.4.1"
RUN python -m pip install -U tensorflow=="2.4.1"
RUN python -m pip install -U torch=="1.7.1+cu110" -f https://download.pytorch.org/whl/torch_stable.html

COPY . /root/Open3D

WORKDIR /root/Open3D
