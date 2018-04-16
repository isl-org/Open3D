file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(IntegrateRGBD ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(IntegrateRGBD PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(IntegrateRGBD Visualization IO Core)
set_target_properties(IntegrateRGBD PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
install(TARGETS IntegrateRGBD
        RUNTIME DESTINATION bin)
