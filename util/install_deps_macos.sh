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
