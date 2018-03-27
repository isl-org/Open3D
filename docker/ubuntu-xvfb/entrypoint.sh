# start virtual framebuffer
# Xvfb :0 -screen 0 800x600x16 -ac &

chmod a+x ~/.xinitrc

# Setup a password
mkdir ~/.vnc
x11vnc -storepasswd 1234 ~/.vnc/passwd

# start vnc
x11vnc -forever -usepw -create
# x11vnc -display :0.0 -forever -usepw
