echo
echo building...

mkdir -p build
cd build

# you can specify a custom install location
cmake ../src -DCMAKE_INSTALL_PREFIX=~/open3d_install

make -j

# this will copy the libs and headers to ~/open3d_install/lib & ~/open3d_install/include
# make install

echo
