#!/usr/bin/env bash
set -e

process_stanford_scene()
{
    DATA_NAME=$1
    GDRIVE_ID=$2
    ./gdrive_download.sh $GDRIVE_ID
    unzip --qq $DATA_NAME"_png.zip" -d ../dataset/stanford/$DATA_NAME
    rm $DATA_NAME"_png.zip"
}

mkdir ../dataset/stanford
process_stanford_scene "burghers" "0B6qjzcYetERgUU0wMkhnZVNCa28"
process_stanford_scene "lounge" "0B6qjzcYetERgSUZFT2FWdWsxQzQ"
process_stanford_scene "copyroom" "0B6qjzcYetERgWTBDYWdkVHN3aHc"
process_stanford_scene "cactusgarden" "0B6qjzcYetERgYUxUSFFIYjZIb3c"
process_stanford_scene "stonewall" "0B6qjzcYetERgOXBCM181bTdsUGc"
process_stanford_scene "totempole" "0B6qjzcYetERgNjVEWm5sSWFlWk0"
