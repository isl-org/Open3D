add_executable(EncodeShader EncodeShader.cpp)
set_target_properties(EncodeShader PROPERTIES FOLDER "Tools")

add_executable(MergeMesh MergeMesh.cpp)
target_link_libraries(MergeMesh IO Core)
set_target_properties(MergeMesh PROPERTIES FOLDER "Tools")

add_executable(ViewGeometry ViewGeometry.cpp)
target_link_libraries(ViewGeometry Visualization IO Core)
set_target_properties(ViewGeometry PROPERTIES FOLDER "Tools")
