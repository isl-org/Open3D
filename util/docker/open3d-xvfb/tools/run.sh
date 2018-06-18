#!/bin/sh

. ./name.sh
./stop.sh

# create shared folder
mkdir -p ~/open3d_docker

# create folder dependencies
mkdir -p ~/open3d_docker/build/lib/Tutorial/Advanced
# cd ~/open3d_docker/build/lib/Tutorial/Advanced

# copy the script for running the docker sample inside a container terminal
cp -f ../setup/docker_sample.sh ~/open3d_docker/build/lib/Tutorial/Advanced
chmod a+x ~/open3d_docker/build/lib/Tutorial/Advanced/docker_sample.sh

# this is the Open3D build script
cp -f ../setup/build.sh ~/open3d_docker
chmod a+x ~/open3d_docker/build.sh

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
       -v ~/open3d_docker:/root/Open3D \
       -p 5920:5900 \
       -h $NAME \
       --name $NAME \
       $NAME

docker container exec -it -w /root/Open3D $NAME bash ./build.sh
