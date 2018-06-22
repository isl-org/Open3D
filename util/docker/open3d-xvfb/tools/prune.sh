#!/bin/sh

./stop.sh

docker container prune
docker image prune
