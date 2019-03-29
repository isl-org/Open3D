#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

# use the name of the upper level directory as the image name
NAME=$(basename $(realpath ${REAL_PATH}/..))

export NAME
