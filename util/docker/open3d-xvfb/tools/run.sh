#!/bin/sh

. ./name.sh

./stop.sh

# create shared folder
mkdir -p ~/Open3D_docker

# create folder dependencies
mkdir -p ~/Open3D_docker/build/lib/Tutorial/Advanced
# cd ~/Open3D_docker/build/lib/Tutorial/Advanced

# copy the headless sample
cp -f ../setup/headless_sample.py ~/Open3D_docker/build/lib/Tutorial/Advanced
chmod a+x ~/Open3D_docker/build/lib/Tutorial/Advanced/headless_sample.py

# copy the script for running the headless sample inside a container terminal
cp -f ../setup/headless_sample.sh ~/Open3D_docker/build/lib/Tutorial/Advanced
chmod a+x ~/Open3D_docker/build/lib/Tutorial/Advanced/headless_sample.sh

# this is the Open3D build script
cp -f ../setup/build.sh ~/Open3D_docker
chmod a+x ~/Open3D_docker/build.sh

# clone into existing non-empty directory
# clone Open3D from the host side, build later inside the container
cd ~/Open3D_docker
git init -q
git remote add origin https://github.com/IntelVCL/Open3D.git
git fetch
git checkout master

# run container with the shared folder as a bind mount
docker container run \
       --rm \
       -d \
       -v ~/Open3D_docker:/root/Open3D \
       -p 5920:5900 \
       -h $NAME \
       --name $NAME \
       $NAME

docker container exec -it -w /root/Open3D $NAME bash ./build.sh
