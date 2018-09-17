echo
echo generating...

cd ../../
mkdir -p build
cd build

# you can specify a custom install location and the python version
cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 -DPYTHON_EXECUTABLE=~/anaconda3/bin/python ..

echo
