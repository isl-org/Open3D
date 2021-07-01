# Gamma Correction example

This example demonstrates gamma correction - a nonlinear operation used to encode and decode the luminance of each image pixel. See https://en.wikipedia.org/wiki/Gamma_correction for more information.

The example creates a fractal image in memory and performs gamma correction on it. The output of the example application is a BMP image with corrected luminance.

|Original image (Y=1)|After applying gamma correction (Y=1.5)|
|---|---|
|<img src="images/original.jpg">|<img src="images/gamma.jpg">|

<br>This example uses C++11 lambda expressions. Specifying a compiler option such as -std=c++11 or similar might be necessary in order to build the example. For more information, please refer to the documentation for the compiler you use.

## Download TBB and build the example using CMake (Linux)
```bash
# Download and unpack TBB 2019 (or newer if available)
export PACKAGE_VER=2019_20181003oss
export TBB_BIN_PACKAGE=tbb${PACKAGE_VER}_lin.tgz
wget https://github.com/01org/tbb/releases/download/2019_U1/${TBB_BIN_PACKAGE}
tar -xzf ${TBB_BIN_PACKAGE}
rm ${TBB_BIN_PACKAGE}
export TBB_INSTALL_DIR=`pwd`/tbb${PACKAGE_VER}
export PSTL_INSTALL_DIR=`pwd`/pstl${PACKAGE_VER}

# Configure, build and run the example
cd ${PSTL_INSTALL_DIR}/examples/gamma_correction
mkdir build && cd build
cmake -DTBB_DIR=$TBB_INSTALL_DIR/cmake -DCMAKE_BUILD_TYPE=Release ..
make && ./gamma_correction
# As a result you'll get two images: image_1.bmp and image_1_gamma.bmp
```
