#!/usr/bin/env bash

mkdir -p ../../../build/open3d_downloads

cd ../../../build/open3d_downloads

wget https://github.com/intel-isl/open3d_downloads/releases/download/icp-examples/kitti_samples_1_499.zip

unzip --q kitti_samples_1_499.zip -d kitti_samples

rm kitti_samples_1_499.zip

