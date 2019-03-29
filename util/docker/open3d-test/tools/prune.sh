#!/bin/bash

docker container prune -f
docker image prune -f

docker image ls
echo
