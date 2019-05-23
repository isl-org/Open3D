#!/bin/sh

. ./name.sh
./stop.sh

Open3D_HOST=~/open3d_docker
Open3D_DOCK=/root/open3d

# create shared folder
mkdir -p $Open3D_HOST

# clone into existing non-empty directory
# clone Open3D from the host side, build later inside the container
cd $Open3D_HOST
git init -q
git remote add origin https://github.com/intel-isl/Open3D.git
git fetch
git checkout master

# run container with the shared folder as a bind mount
docker container run \
       --rm \
       -d \
       -v $Open3D_HOST:$Open3D_DOCK \
       -p 5920:5900 \
       -h $NAME \
       --name $NAME \
       $NAME

docker container exec -it -w $Open3D_DOCK $NAME bash -c 'mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=~/open3d_install && make -j && make install && bash'
