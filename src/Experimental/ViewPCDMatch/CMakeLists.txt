file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ViewPCDMatch ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(ViewPCDMatch Visualization IO Core)
set_target_properties(ViewPCDMatch PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
