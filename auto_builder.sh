#!/bin/bash

set -eu

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/build

cd ${DIR}
make unitTests -j
./bin/unitTests --gtest_filter="*SumCUDA*"
cd ..
