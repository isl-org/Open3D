#!/bin/sh

# Setup a password
mkdir ~/.vnc
x11vnc -storepasswd 1234 ~/.vnc/passwd

# start vnc
x11vnc -forever -usepw -create
