echo
echo cleaning...

rm -rf build

# remove the CMake cache
find . -name CMakeFiles -type d -exec rm -rf {} +
find . -name CMakeCache.txt -exec rm -rf {} +
find . -name cmake_install.cmake -exec rm -rf {} +

# Install the project...
# -- Install configuration: "Release"
# -- Installing: /home/dpetre/.local/lib/Open3D/libCore.a
# -- Installing: /home/dpetre/.local/include/Open3D/Eigen.h
# -- Installing: /home/dpetre/.local/include/Open3D/Helper.h
# -- Installing: /home/dpetre/.local/include/Open3D/Timer.h
# -- Installing: /home/dpetre/.local/include/Open3D/FileSystem.h
# -- Installing: /home/dpetre/.local/include/Open3D/Console.h
# -- Installing: /home/dpetre/.local/lib/Open3D/libIO.a
# -- Installing: /home/dpetre/.local/lib/Open3D/libVisualization.a
# -- Installing: /home/dpetre/.local/include/Open3D/RenderOptionWithEditing.h
# -- Installing: /home/dpetre/.local/include/Open3D/RenderOption.h
# -- Installing: /home/dpetre/.local/include/Open3D/Visualizer.h
# -- Installing: /home/dpetre/.local/include/Open3D/ViewControlWithCustomAnimation.h
# -- Installing: /home/dpetre/.local/include/Open3D/VisualizerWithKeyCallback.h
# -- Installing: /home/dpetre/.local/include/Open3D/VisualizerWithEditing.h
# -- Installing: /home/dpetre/.local/include/Open3D/ViewControl.h
# -- Installing: /home/dpetre/.local/include/Open3D/ViewControlWithEditing.h
# -- Installing: /home/dpetre/.local/include/Open3D/ViewParameters.h
# -- Installing: /home/dpetre/.local/include/Open3D/ViewTrajectory.h
# -- Installing: /home/dpetre/.local/include/Open3D/VisualizerWithCustomAnimation.h

# remove the Open3D runtime binaries
rm -rf ~/.local/bin/Open3D

# remove the Open3D libs
rm -rf ~/.local/lib/Open3D

# remove the Open3D headers
rm -rf ~/.local/include/Open3D

echo
