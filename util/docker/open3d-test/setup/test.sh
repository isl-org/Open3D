#!/bin/sh

echo

git clone --recurse-submodules https://github.com/IntelVCL/Open3D.git open3d
cd open3d
echo

echo "building STATIC..."
date

mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE=/usr/bin/${PYTHON} \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_UNIT_TESTS=ON
echo
make -j
date
echo

echo building pip package...
date
make pip-package
#make install-pip-package
date
echo

echo running the unit tests...
./bin/unitTests

date
echo

echo cleaning...
cd ..
rm -rf build
echo

echo building SHARED...
date

mkdir -p build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON \
         -DPYTHON_EXECUTABLE=/usr/bin/${PYTHON} \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_UNIT_TESTS=ON
echo
make -j
date
echo

echo building pip package...
date
make pip-package
# make install-pip-package
date
echo

echo running the unit tests...
./bin/unitTests

date
echo