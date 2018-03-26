
# Setup a password
mkdir ~/.vnc
x11vnc -storepasswd 1234 ~/.vnc/passwd

# start virtual framebuffer
# Xvfb -screen 0 800x600x32 -ac &

# start ratpoison
# ratpoison &

# start vnc
x11vnc -forever -usepw -create