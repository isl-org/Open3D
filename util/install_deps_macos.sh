#!/usr/bin/env bash
set -e

which -s brew
if [[ $? != 0 ]]; then
    echo "Please install Homebrew, follow the instructions on:"
    echo "        http://brew.sh/"
    echo "After installation, run this script again."
    exit
else
    echo "Homebrew Detected."
    echo "Performing update ..."
    if [ "$1" == "skip-upgrade" ]; then
        echo "brew update skipped."
    else
        brew update  # `brew update` upgrades brew itself.
    fi
fi

for pkg in libusb pkg-config tbb; do
    if brew list -1 | grep -q "^${pkg}\$"; then
        if [ "$1" == "skip-upgrade" ]; then
            echo "Package '$pkg' has already been installed."
        else
            echo "Package '$pkg' has already been installed and is being upgraded ..."
            # Using HOMEBREW_NO_AUTO_UPDATE=1 since `brew update` has already
            # been performed if needed. Upgrade might cause error when already
            # installed and thus `|| true` is used.
            HOMEBREW_NO_AUTO_UPDATE=1 brew upgrade $pkg || true
        fi
    else
        echo "Package '$pkg' is being installed ..."
        HOMEBREW_NO_AUTO_UPDATE=1 brew install $pkg
    fi
done
