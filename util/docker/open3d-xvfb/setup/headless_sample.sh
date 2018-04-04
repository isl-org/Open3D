Xvfb :99 -screen 0 1920x1080x24 &
DISPLAY=:99.0
export DISPLAY

./headless_sample.py