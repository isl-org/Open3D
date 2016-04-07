#!/bin/sh

which -s brew
if [[ $? != 0 ]] ; then
    echo "Install Homebrew."
    echo "Please follow the instructions on http://brew.sh/"
    echo "After installation, run this script again."
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
