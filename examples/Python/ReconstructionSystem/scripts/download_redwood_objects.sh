#!/usr/bin/env bash
set -e

process_redwood_object()
{
    DATA_NUM=$1
    DATA_NAME=$2
    GDRIVE_ID=$3
    ./gdrive_download.sh $GDRIVE_ID
    unzip --qq $DATA_NUM.zip -d ../dataset/redwood_objects/$DATA_NAME
    python synchronize_frames.py ${PWD}/../dataset/redwood_objects/$DATA_NAME
    rm $DATA_NUM.zip
}

mkdir ../dataset/redwood_objects
process_redwood_object "00021" "chair" "1iMxjIZMFcoL3s9FzzqM0K-tM2ehO74D0"
process_redwood_object "00577" "sofa" "1_WZK0AZTt7N3QBh9JnDypM1sPKnE0oIY"
process_redwood_object "01833" "car" "1HCRVzlZ0huAsTyQsjOvI2OsZ_oU3zpt9"
process_redwood_object "05984" "motorcycle" "1jZLTNrOIP2sFgzF9sHb027iI2JhOOqpu"
process_redwood_object "06127" "plant" "1nWilJfkAA7D3a8JEc_Tx9pLD4a25u8xG"
process_redwood_object "06822" "truck" "1TqkWcdzQZG50ZV9nXdZYZmR_aLH-WYkr"
