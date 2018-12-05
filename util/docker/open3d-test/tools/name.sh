#!/bin/sh

# use the name of the upper level directory as the image name
NAME=$(bash -c 'basename $(cd .. ; pwd)')

export NAME
