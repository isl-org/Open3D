file(GLOB HEADER_FILES "*.h")
file(GLOB SOURCE_FILES "*.cpp")
add_executable(RenderDistance ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(RenderDistance Visualization IO Core)
set_target_properties(RenderDistance PROPERTIES
		FOLDER "Sandbox"
	RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Sandbox")
