echo
echo building...

mkdir -p build
cd build

# you can specify a custom install location and the python version
cmake -DCMAKE_INSTALL_PREFIX=~/open3d_install .. #-DPYTHON_EXECUTABLE=/usr/bin/python3.5

make -j

# this will copy the libs and headers to ~/open3d_install/lib & ~/open3d_install/include
# make install

echo
