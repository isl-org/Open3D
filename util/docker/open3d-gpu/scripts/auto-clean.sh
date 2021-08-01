#!/usr/bin/env bash

# https://cloud.google.com/community/tutorials/create-a-self-deleting-virtual-machine
sleep 20m
export NAME=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google')
export ZONE=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google')
gcloud --quiet compute instances delete $NAME --zone=$ZONE
