echo
echo generating...

mkdir -p build
cd build

cmake ../src

# you can specify a custom install location
# cmake ../src/ -DCMAKE_INSTALL_PREFIX=~/Open3D_install

echo
