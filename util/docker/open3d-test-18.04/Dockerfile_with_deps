FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

# install Open3D dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libjpeg-dev \
    libjsoncpp-dev \
    libosmesa6-dev \
    libpng-dev \
    mesa-utils \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    pybind11-dev \
    software-properties-common \
    xorg-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install matplotlib numpy

WORKDIR /root

COPY ./setup/install-gtest.sh .
RUN chmod a+x /root/install-gtest.sh
RUN /root/install-gtest.sh
