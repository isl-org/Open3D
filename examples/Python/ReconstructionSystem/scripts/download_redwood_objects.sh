#!/bin/sh
process_redwood_object()
{
    DATA_NUM=$1
    DATA_NAME=$2
    unzip --qq $DATA_NUM.zip -d ../dataset/redwood_objects/$DATA_NAME
    python synchronize_frames.py ${PWD}/../dataset/redwood_objects/$DATA_NAME
}

./gdrive_download.sh 1iMxjIZMFcoL3s9FzzqM0K-tM2ehO74D0
./gdrive_download.sh 1_WZK0AZTt7N3QBh9JnDypM1sPKnE0oIY
./gdrive_download.sh 1HCRVzlZ0huAsTyQsjOvI2OsZ_oU3zpt9
./gdrive_download.sh 1jZLTNrOIP2sFgzF9sHb027iI2JhOOqpu
./gdrive_download.sh 1nWilJfkAA7D3a8JEc_Tx9pLD4a25u8xG
./gdrive_download.sh 1TqkWcdzQZG50ZV9nXdZYZmR_aLH-WYkr

mkdir ../dataset/redwood_objects
process_redwood_object "00021" "chair"
process_redwood_object "00577" "sofa"
process_redwood_object "01833" "car"
process_redwood_object "05984" "motorcycle"
process_redwood_object "06127" "plant"
process_redwood_object "06822" "truck"

rm *.zip
