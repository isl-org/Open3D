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
    gdb \
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
    lxde \
    mesa-utils \
    ne \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    python3-dbg \
    pybind11-dev \
    software-properties-common \
    x11vnc \
    xorg-dev \
    xterm \
    xvfb \
    wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install \
    matplotlib \
    numpy \
    opencv-python

EXPOSE 5920

ENV HOME=/root

WORKDIR $HOME
COPY ./setup/.xinitrc .
RUN chmod a+x .xinitrc
COPY ./setup/entrypoint.sh .

COPY ./setup/install-gtest.sh .
RUN chmod a+x /root/install-gtest.sh
RUN /root/install-gtest.sh

# CMD ["bash", "/root/entrypoint.sh"]
