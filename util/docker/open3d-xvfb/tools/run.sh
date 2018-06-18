#!/bin/sh

. ./name.sh
./stop.sh

# create shared folder
mkdir -p ~/open3d_docker

# clone into existing non-empty directory
# clone Open3D from the host side, build later inside the container
cd ~/open3d_docker
git init -q
git remote add origin https://github.com/IntelVCL/Open3D.git
git fetch
git checkout master

# run container with the shared folder as a bind mount
docker container run \
       --rm \
       -d \
       -v ~/open3d_docker:/root/open3d \
       -p 5920:5900 \
       -h $NAME \
       --name $NAME \
       $NAME

docker container exec -it -w /root/open3d $NAME bash -c 'mkdir -p build && cd build && cmake ../src -DBUILD_PYBIND11=OFF -DCMAKE_INSTALL_PREFIX=~/open3d_install && make -j && make install && bash'
