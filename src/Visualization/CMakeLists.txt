# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "Shader/*.glsl")
add_custom_command(
	OUTPUT ${Open3DV_SOURCE_DIR}/Visualization/Shader.h
	COMMAND EncodeShader Shader.h
	WORKING_DIRECTORY ${Open3DV_SOURCE_DIR}/Visualization/
	DEPENDS ${SHADER_FILES} EncodeShader
)
foreach(file ${SHADER_FILES})
	add_custom_command(
		OUTPUT ${Open3DV_SOURCE_DIR}/Visualization/Shader.h
		COMMAND EncodeShader Shader.h ${file}
		WORKING_DIRECTORY ${Open3DV_SOURCE_DIR}/Visualization/
		DEPENDS ${SHADER_FILES} EncodeShader
		APPEND
	)
endforeach(file)
add_custom_target(ShaderFileTarget ALL
	DEPENDS ${Open3DV_SOURCE_DIR}/Visualization/Shader.h)
set_target_properties(ShaderFileTarget PROPERTIES FOLDER "Custom")

source_group("Shader" FILES ${SHADER_FILES})
file(GLOB VISUALIZATION_GLEW_FILES
	"../External/glew/include/GL/*.h"
	"../External/glew/src/glew.c"
)
source_group("glew" FILES ${VISUALIZATION_GLEW_FILES})
file(GLOB VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB VISUALIZATION_ALL_SOURCE_FILES "*.cpp")
add_library(Visualization ${Open3DV_LIB_TYPE}
	${VISUALIZATION_ALL_SOURCE_FILES}
	${VISUALIZATION_ALL_HEADER_FILES}
	${VISUALIZATION_GLEW_FILES}
	${SHADER_FILES}
)
add_dependencies(Visualization ShaderFileTarget)

install(TARGETS Visualization
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
