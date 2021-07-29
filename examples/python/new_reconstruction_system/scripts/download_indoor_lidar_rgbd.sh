#!/usr/bin/env bash
set -e

process_scene()
{
    DATA_NAME=$1
    GDRIVE_ID=$2
    ./gdrive_download.sh $GDRIVE_ID
    unzip --qq $DATA_NAME.zip -d ../dataset/indoor_lidar_rgbd
    rm $DATA_NAME.zip
}

mkdir ../dataset/indoor_lidar_rgbd
process_scene "rgbd_apartment" "1TqoIE1Q1M30q8FBQ_dMcZj9JR6G0ztI5"
process_scene "rgbd_bedroom" "1SN318pHOQn7ioSABJLY6SQ1O7gVMDvB2"
process_scene "rgbd_boardroom" "1gRDVGgPR--cKKHkrlaXVzc1Zj9VhC3Dr"
process_scene "rgbd_lobby" "1MhjCJuznp3pG6zxHrIbmcjPjXhvlBStu"
process_scene "rgbd_loft" "1OOmymidV5nhmGSdk1Y7yI_9fXxLHNPjX"
