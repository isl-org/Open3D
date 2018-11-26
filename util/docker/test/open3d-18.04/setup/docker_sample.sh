#!/bin/sh

Xvfb :99 -screen 0 1920x1080x24 &
DISPLAY=:99.0
export DISPLAY

python3 ./headless_rendering.py
