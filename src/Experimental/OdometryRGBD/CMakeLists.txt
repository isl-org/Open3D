file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(OdometryRGBD ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(OdometryRGBD IO Core)
set_target_properties(OdometryRGBD PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
