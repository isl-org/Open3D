file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ManuallyAlignPointCloud ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ManuallyAlignPointCloud Visualization IO Core)
set_target_properties(ManuallyAlignPointCloud PROPERTIES
	FOLDER "Tools")
