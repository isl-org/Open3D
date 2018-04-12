add_subdirectory(ManuallyAlignPointCloud)

project(EncodeShader)
add_executable(${PROJECT_NAME} EncodeShader.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Tools")

project(MergeMesh)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} IO Core)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Tools")

project(ConvertPointCloud)
add_executable(${PROJECT_NAME} ConvertPointCloud.cpp)
target_link_libraries(${PROJECT_NAME} IO Core)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Tools")

project(ManuallyCropPointCloud)
add_executable(${PROJECT_NAME} ManuallyCropPointCloud.cpp)
target_link_libraries(${PROJECT_NAME} Visualization IO Core)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Tools")

project(ViewGeometry)
add_executable(${PROJECT_NAME} ViewGeometry.cpp)
target_link_libraries(${PROJECT_NAME} Visualization IO Core)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "Tools")
