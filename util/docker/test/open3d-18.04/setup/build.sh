echo
echo building...

git clone https://github.com/IntelVCL/Open3D.git open3d
cd open3d

mkdir -p build

# Custom install location:              -DCMAKE_INSTALL_PREFIX=~/open3d_install
# Build unit test:                      -DBUILD_UNIT_TESTS=ON
# Specify the python version:           -DPYTHON_EXECUTABLE=/usr/bin/python3.5
# Disable building the pythonb module:  -DBUILD_PYTHON_MODULE=OFF
# Specify the build type:               -DCMAKE_BUILD_TYPE=Release

cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/open3d_install -DBUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release

echo

make -j

echo

# copy the libs to:     ~/open3d_install/lib
# copy the headers to:  ~/open3d_install/include
make install

# install the python module using pip
# make install-pip-package

echo
