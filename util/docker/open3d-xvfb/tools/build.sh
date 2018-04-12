#!/bin/sh

. ./name.sh
./stop.sh

docker image build -t $NAME ..
