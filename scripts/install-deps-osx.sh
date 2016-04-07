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
    sudo brew update
fi

for pkg in cmake libusb; do
    if brew list -1 | grep -q "^${pkg}\$"; then
        echo "Package '$pkg' has already been installed."
    else
        echo "Package '$pkg' is being installed ..."
        brew install $pkg
    fi
done
