rm -rf build-xcode
mkdir build-xcode
cd build-xcode
cmake .. -G Xcode -DCMAKE_INSTALL_PREFIX=~/open3d_install/
open Open3D.xcodeproj
