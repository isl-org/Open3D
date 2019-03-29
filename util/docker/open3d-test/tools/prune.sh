#!/usr/bin/env bash
set -e

docker container prune -f
docker image prune -f

docker image ls
echo
