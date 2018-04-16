file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(EvaluateFeatureMatch ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(EvaluateFeatureMatch Visualization IO Core)
set_target_properties(EvaluateFeatureMatch PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
