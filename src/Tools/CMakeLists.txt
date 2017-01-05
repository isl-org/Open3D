add_subdirectory(ManuallyAlignPointCloud)

add_executable(EncodeShader EncodeShader.cpp)
set_target_properties(EncodeShader PROPERTIES FOLDER "Tools")

add_executable(MergeMesh MergeMesh.cpp)
target_link_libraries(MergeMesh IO Core)
set_target_properties(MergeMesh PROPERTIES FOLDER "Tools")

add_executable(ConvertPointCloud ConvertPointCloud.cpp)
target_link_libraries(ConvertPointCloud IO Core)
set_target_properties(ConvertPointCloud PROPERTIES FOLDER "Tools")

add_executable(ManuallyCropPointCloud ManuallyCropPointCloud.cpp)
target_link_libraries(ManuallyCropPointCloud Visualization IO Core)
set_target_properties(ManuallyCropPointCloud PROPERTIES FOLDER "Tools")

add_executable(ViewGeometry ViewGeometry.cpp)
target_link_libraries(ViewGeometry Visualization IO Core)
set_target_properties(ViewGeometry PROPERTIES FOLDER "Tools")
