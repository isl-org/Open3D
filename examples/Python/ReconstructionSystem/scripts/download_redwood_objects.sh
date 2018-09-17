#!/bin/sh
unzip_redwood_scene()
{
    DATA_NUM=$1
    DATA_NAME=$2
    mkdir ../dataset/redwood_objects/$DATA_NUM
    unzip --qq $DATA_NUM.zip -d ../dataset/redwood_objects/$DATA_NAME
}

./gdrive_download.sh 1iMxjIZMFcoL3s9FzzqM0K-tM2ehO74D0
./gdrive_download.sh 1_WZK0AZTt7N3QBh9JnDypM1sPKnE0oIY
./gdrive_download.sh 1HCRVzlZ0huAsTyQsjOvI2OsZ_oU3zpt9
./gdrive_download.sh 1jZLTNrOIP2sFgzF9sHb027iI2JhOOqpu
./gdrive_download.sh 1nWilJfkAA7D3a8JEc_Tx9pLD4a25u8xG
./gdrive_download.sh 1TqkWcdzQZG50ZV9nXdZYZmR_aLH-WYkr

mkdir ../dataset/redwood_objects
unzip_redwood_scene "00021" "chair"
unzip_redwood_scene "00577" "sofa"
unzip_redwood_scene "01833" "car"
unzip_redwood_scene "05984" "motorcycle"
unzip_redwood_scene "06127" "plant"
unzip_redwood_scene "06822" "truck"

rm *.zip
