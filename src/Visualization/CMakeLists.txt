# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "Shader/GLSL/*.glsl")
add_custom_command(
	OUTPUT ${Open3D_SOURCE_DIR}/Visualization/Shader/Shader.h
	COMMAND EncodeShader Shader.h
	WORKING_DIRECTORY ${Open3D_SOURCE_DIR}/Visualization/Shader/
	DEPENDS ${SHADER_FILES} EncodeShader
)
foreach(file ${SHADER_FILES})
	add_custom_command(
		OUTPUT ${Open3D_SOURCE_DIR}/Visualization/Shader/Shader.h
		COMMAND EncodeShader Shader.h ${file}
		WORKING_DIRECTORY ${Open3D_SOURCE_DIR}/Visualization/Shader/
		DEPENDS ${SHADER_FILES} EncodeShader
		APPEND
	)
endforeach(file)
add_custom_target(ShaderFileTarget ALL
DEPENDS ${Open3D_SOURCE_DIR}/Visualization/Shader/Shader.h)
set_target_properties(ShaderFileTarget PROPERTIES FOLDER "Custom")

source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})
file(GLOB VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB VISUALIZATION_ALL_SOURCE_FILES "*.cpp")
file(GLOB VISUALIZATION_UTILITY_HEADER_FILES "Utility/*.h")
source_group("Header Files\\Utility" FILES ${VISUALIZATION_UTILITY_HEADER_FILES})
file(GLOB VISUALIZATION_UTILITY_SOURCE_FILES "Utility/*.cpp")
source_group("Source Files\\Utility" FILES ${VISUALIZATION_UTILITY_SOURCE_FILES})
file(GLOB VISUALIZATION_SHADER_HEADER_FILES "Shader/*.h")
source_group("Header Files\\Shader" FILES ${VISUALIZATION_SHADER_HEADER_FILES})
file(GLOB VISUALIZATION_SHADER_SOURCE_FILES "Shader/*.cpp")
source_group("Source Files\\Shader" FILES ${VISUALIZATION_SHADER_SOURCE_FILES})
file(GLOB VISUALIZATION_VISUALIZER_HEADER_FILES "Visualizer/*.h")
source_group("Header Files\\Visualizer" FILES ${VISUALIZATION_VISUALIZER_HEADER_FILES})
file(GLOB VISUALIZATION_VISUALIZER_SOURCE_FILES "Visualizer/*.cpp")
source_group("Source Files\\Visualizer" FILES ${VISUALIZATION_VISUALIZER_SOURCE_FILES})

add_library(Visualization ${Open3D_LIB_TYPE}
	${SHADER_FILES}
	${VISUALIZATION_ALL_HEADER_FILES} ${VISUALIZATION_ALL_SOURCE_FILES}
	${VISUALIZATION_UTILITY_HEADER_FILES} ${VISUALIZATION_UTILITY_SOURCE_FILES}
	${VISUALIZATION_SHADER_HEADER_FILES} ${VISUALIZATION_SHADER_SOURCE_FILES}
	${VISUALIZATION_VISUALIZER_HEADER_FILES} ${VISUALIZATION_VISUALIZER_SOURCE_FILES}
)
add_dependencies(Visualization ShaderFileTarget)
target_link_libraries(Visualization Core ${GLFW_LIBRARIES} ${GLEW_LIBRARIES})

install(TARGETS Visualization
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
