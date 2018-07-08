#!/bin/sh

which -s brew
if [[ $? != 0 ]] ; then
    echo "Please install Homebrew, follow the instructions on:"
    echo ""
    echo "        http://brew.sh/"
    echo ""
    echo "After installation, run this script again."
    exit
else
    echo "Homebrew Detected."
    echo "Performing update ..."
    brew update
fi

for pkg in cmake libusb glew glfw3 libpng jpeg pkg-config jsoncpp eigen; do
    if brew list -1 | grep -q "^${pkg}\$"; then
        echo "Package '$pkg' has already been installed and is being upgraded ..."
        brew upgrade $pkg
    else
        echo "Package '$pkg' is being installed ..."
        brew install $pkg
    fi
done
