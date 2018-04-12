#!/bin/sh

. ./name.sh
./stop.sh

docker image rm $NAME:latest
