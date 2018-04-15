file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(ViewPCDMatch ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(ViewPCDMatch PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(ViewPCDMatch Visualization IO Core)
set_target_properties(ViewPCDMatch PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
install(TARGETS ViewPCDMatch
        RUNTIME DESTINATION bin)
