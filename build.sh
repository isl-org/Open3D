echo
echo building...

mkdir -p build
cd build

# you can specify a custom install location
cmake ../src -DCMAKE_INSTALL_PREFIX=~/open3d_install #-DOpen3D_USE_NATIVE_DEPENDENCY_BUILD=OFF

make -j

# this will copy the libs and headers to ~/.local/lib & ~/.local/include
# make install

echo
