file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(EvaluatePCDMatch ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(EvaluatePCDMatch Visualization IO Core)
set_target_properties(EvaluatePCDMatch PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
