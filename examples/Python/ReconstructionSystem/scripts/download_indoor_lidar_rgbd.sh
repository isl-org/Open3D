#!/bin/sh
cd .. && mkdir dataset/indoor_lidar_rgbd && cd dataset/indoor_lidar_rgbd
../../scripts/gdrive_download.sh 1TqoIE1Q1M30q8FBQ_dMcZj9JR6G0ztI5
../../scripts/gdrive_download.sh 1SN318pHOQn7ioSABJLY6SQ1O7gVMDvB2
../../scripts/gdrive_download.sh 1gRDVGgPR--cKKHkrlaXVzc1Zj9VhC3Dr
../../scripts/gdrive_download.sh 1MhjCJuznp3pG6zxHrIbmcjPjXhvlBStu
../../scripts/gdrive_download.sh 1OOmymidV5nhmGSdk1Y7yI_9fXxLHNPjX
unzip --qq rgbd_apartment.zip 
unzip --qq rgbd_boardroom.zip
unzip --qq rgbd_lobby.zip
unzip --qq rgbd_loft.zip
unzip --qq rgbd_bedroom.zip
rm *.zip
