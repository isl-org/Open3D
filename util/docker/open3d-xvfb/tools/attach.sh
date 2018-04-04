#!/bin/sh

. ./name.sh

./run.sh

docker container exec -it $NAME bash
