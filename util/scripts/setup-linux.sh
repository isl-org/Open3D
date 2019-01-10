#!/bin/bash

pwd

./util/scripts/install-deps-ubuntu.sh
./util/scripts/make-documentation.sh
./util/scripts/install-gtest.sh

python --version
cmake --version
