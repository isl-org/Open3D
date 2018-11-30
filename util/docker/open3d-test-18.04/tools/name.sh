#!/bin/sh

# get the name of the upper level directory
NAME=$(bash -c 'basename $(cd .. ; pwd)')

export NAME
