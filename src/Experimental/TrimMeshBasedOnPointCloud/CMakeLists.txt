file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(TrimMeshBasedOnPointCloud ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(TrimMeshBasedOnPointCloud Visualization IO Core)
set_target_properties(TrimMeshBasedOnPointCloud PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
