#!/usr/bin/env bash

# Build Mesa CPU rendering library libGL.so following instructions from
# https://docs.mesa3d.org/install.html
#
# To build on Ubuntu, simply run this script. sudo access is needed and it will
# modify your system. Alternately, you can build in an isolated docker container
# as follows:
#
# cd Open3D/3rdparty/mesa
# docker run --rm -v $PWD:/host_dir --entrypoint /host_dir/build-mesa-cpu.sh \
# --workdir /root/ ubuntu:bionic
# sudo chown $(id -u):$(id -g) libGL.so.1.5.0
#
# This will build libGL.so.1.5 in the docker container and copy the result back
# to the current directory.

MESA_VER=22.0  # latest tag on 2022-03-10
[ "$EUID" -ne 0 ] && SUDO="sudo" || SUDO="command"

echo "Enable source repositories and get build dependencies."
$SUDO sed -i '/deb-src/s/^# //' /etc/apt/sources.list &&
$SUDO apt-get update &&
$SUDO apt-get --yes build-dep mesa &&
$SUDO apt-get --yes install wget
echo "Check and update meson (Ubuntu 18.04 has a too old meson.)"
if dpkg --compare-versions "$(meson -v)" lt 0.46.1 ; then
    $SUDO apt-get --yes install python3-pip
    yes | pip3 install "meson>=0.46.1,<0.62.0"  # 0.62.0 disables Python 3.6 suport
    export PATH=/usr/local/bin:$PATH  # meson from pip before system meson
fi
echo "Disable source repositories again."
$SUDO sed -i '/deb-src/s/^/# /' /etc/apt/sources.list

echo "Get Mesa source code version $MESA_VER"
wget -c \
https://gitlab.freedesktop.org/mesa/mesa/-/archive/${MESA_VER}/mesa-${MESA_VER}.tar.bz2 \
-O - | tar -xj
pushd mesa-${MESA_VER}
echo Configure...
meson build/ \
    `# X11 SW rendering` \
    -Dglx=xlib -Dgallium-drivers=swrast -Dplatforms=x11 \
    `# Disable HW drivers` \
    -Ddri3=false -Ddri-drivers= -Dvulkan-drivers= \
    -Dlmsensors=disabled \
    `# Optimization, remove debug info` \
    -Dbuildtype=release -Doptimization=3 -Db_lto=true -Dstrip=true \
    `# Security hardening` \
    -Dcpp_args="-D_FORTIFY_SOURCE=2 -fstack-protector-strong -Wformat -Wformat-security" \
    -Dcpp_link_args="-Wl,-z,noexecstack -Wl,-z,relro,-z,now"

echo Build...
ninja -C build/
echo "Copy libGL.so out"
[ -d /host_dir ] && OUT_DIR="/host_dir" || OUT_DIR=".."
cp build/src/gallium/targets/libgl-xlib/libGL.so.1.5.0 $OUT_DIR
popd
