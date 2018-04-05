echo
echo cleaning...

rm -rf build

# remove the CMake cache
find . -name CMakeFiles -type d -exec rm -rf {} +
find . -name CMakeCache.txt -exec rm -rf {} +
find . -name cmake_install.cmake -exec rm -rf {} +

# Install the project...
# -- Install configuration: "Release"
# -- Installing: /home/dpetre/.local/lib/libCore.a
# -- Installing: /home/dpetre/.local/include/Core.h
# -- Installing: /home/dpetre/.local/lib/libIO.a
# -- Installing: /home/dpetre/.local/include/IO.h
# -- Installing: /home/dpetre/.local/lib/libVisualization.a
# -- Installing: /home/dpetre/.local/include/Visualization.h
# -- Installing: /home/dpetre/.local/lib/python3.5/site-packages/py3d.cpython-35m-x86_64-linux-gnu.so

# remove the Open3D libs
rm -rf ~/.local/lib/libCore.a
rm -rf ~/.local/lib/libIO.a
rm -rf ~/.local/lib/libVisualization.a

# remove the Open3D headers
rm -rf ~/.local/include/Core.h
rm -rf ~/.local/include/IO.h
rm -rf ~/.local/include/Visualization.h

# remove the Open3D Python module
rm -rf ~/.local/lib/python3.5/site-packages/py3d.cpython-35m-x86_64-linux-gnu.so

echo
