#!/usr/bin/env bash
set -e

download_redwood_scene()
{
    DATA_NAME=$1
    wget http://redwood-data.org/indoor/data/$DATA_NAME-color.zip
    wget http://redwood-data.org/indoor/data/$DATA_NAME-depth-clean.zip
    wget http://redwood-data.org/indoor/data/$DATA_NAME-depth-simulated.zip
}

unzip_redwood_scene()
{
    DATA_NAME=$1
    DATA_TYPE=$2
    mkdir $DATA_NAME-$DATA_TYPE
    mkdir $DATA_NAME-$DATA_TYPE/image
    mkdir $DATA_NAME-$DATA_TYPE/depth
    unzip --qq $DATA_NAME-color.zip -d $DATA_NAME-$DATA_TYPE/image
    unzip --qq $DATA_NAME-depth-$DATA_TYPE.zip -d $DATA_NAME-$DATA_TYPE/depth
}

cd .. && mkdir dataset/redwood_simulated && cd dataset/redwood_simulated

download_redwood_scene "livingroom1"
unzip_redwood_scene "livingroom1" "clean"
# unzip_redwood_scene "livingroom1" "simulated"

download_redwood_scene "livingroom2"
unzip_redwood_scene "livingroom2" "clean"
# unzip_redwood_scene "livingroom2" "simulated"

download_redwood_scene "office1"
unzip_redwood_scene "office1" "clean"
# unzip_redwood_scene "office1" "simulated"

download_redwood_scene "office2"
unzip_redwood_scene "office2" "clean"
# unzip_redwood_scene "office2" "simulated"

rm *.zip
