echo
echo building...

mkdir -p build
cd build

cmake ../src

# you can specify a custom install location
# cmake ../src/ -DCMAKE_INSTALL_PREFIX=~/Open3D_install

make -j

# this will copy the static libs and corresponding headers to ~/.local/lib & ~/.local/include
make install

echo
