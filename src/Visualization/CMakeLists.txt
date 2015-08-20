# some black magic to automatically create a header file from shader files
add_custom_command(
	OUTPUT Shader.h
	COMMAND EncodeShader Shader.h
	WORKING_DIRECTORY ${Open3DV_SOURCE_DIR}/Visualization/
)
set(ShaderOutputFiles "Shader.h")
file(GLOB SHADER_FILES "shader/*.glsl")
foreach(file ${SHADER_FILES})
	add_custom_command(
		OUTPUT "${file}.out"
		COMMAND EncodeShader Shader.h ${file}
		WORKING_DIRECTORY ${Open3DV_SOURCE_DIR}/Visualization/
	)
	set(ShaderOutputFiles ${ShaderOutputFiles} "${file}.out")
endforeach(file)
add_custom_target(ShaderFileTarget ALL DEPENDS ${ShaderOutputFiles})

source_group("Shader" FILES ${SHADER_FILES})
file(GLOB VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB VISUALIZATION_ALL_SOURCE_FILES "*.cpp")
add_library(Visualization ${Open3DV_LIB_TYPE}
	${VISUALIZATION_ALL_SOURCE_FILES}
	${VISUALIZATION_ALL_HEADER_FILES}
	${SHADER_FILES}
)

install(TARGETS Visualization
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
