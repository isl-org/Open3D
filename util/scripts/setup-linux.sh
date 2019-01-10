#!/bin/bash

./install-deps-ubuntu.sh
./make-documentation.sh
./install-gtest.sh

python --version
cmake --version
