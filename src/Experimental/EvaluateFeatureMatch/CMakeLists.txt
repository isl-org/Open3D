file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(EvaluateFeatureMatch ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(EvaluateFeatureMatch PRIVATE ${EIGEN3_INCLUDE_DIR} ${FLANN_INCLUDE_DIR})
target_link_libraries(EvaluateFeatureMatch Visualization IO Core ${FLANN_CPP_LIBRARY_STATIC})
set_target_properties(EvaluateFeatureMatch PROPERTIES
		FOLDER "Experimental"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Experimental")
install(TARGETS EvaluateFeatureMatch
        RUNTIME DESTINATION bin)
