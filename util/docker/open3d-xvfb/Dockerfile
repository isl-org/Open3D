FROM ubuntu:16.04

# ENV http_proxy  ""
# ENV HTTP_PROXY  ""
# ENV https_proxy ""
# ENV HTTPS_PROXY ""
# ENV ftp_proxy   ""
# ENV socks_proxy ""
# ENV no_proxy    ""
# ENV ALL_PROXY   ""

ENV DEBIAN_FRONTEND noninteractive

# install Open3D dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends\
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

WORKDIR $HOME/Open3D/build/lib/Tutorial/Advanced

CMD ["bash", "/root/entrypoint.sh"]
