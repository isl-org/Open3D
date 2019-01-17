#!/bin/bash

set -e

./util/scripts/install-deps-ubuntu.sh
./util/scripts/make-documentation.sh
./util/scripts/install-gtest.sh
