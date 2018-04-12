file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
project(ManuallyAlignPointCloud)
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Visualization IO Core)
set_target_properties(${PROJECT_NAME} PROPERTIES
	FOLDER "Tools")
