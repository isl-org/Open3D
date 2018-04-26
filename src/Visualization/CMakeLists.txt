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

# build
file(GLOB_RECURSE VISUALIZATION_ALL_SOURCE_FILES "*.cpp")

include_directories(".")
include_directories("/Shader")
include_directories("/Utility")
include_directories("/Visualizer")

add_library(Visualization
	        ${SHADER_FILES}
            ${VISUALIZATION_ALL_SOURCE_FILES})

add_dependencies(Visualization ShaderFileTarget)
target_link_libraries(Visualization
                      Core
                      IO
                      ${GLFW_LIBRARIES}
                      ${GLEW_LIBRARIES})

if (Open3D_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    include_directories(${OSMESA_INCLUDE_DIR})
    target_link_libraries(Visualization ${OSMESA_LIBRARY})
else ()
    find_package(OpenGL REQUIRED)
    target_link_libraries(Visualization ${OPENGL_LIBRARIES})
endif ()

#install
install(TARGETS Visualization
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS_ROOT       "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB HEADERS_SHADER     "${CMAKE_CURRENT_SOURCE_DIR}/Shader/*.h")
file(GLOB HEADERS_UTILITY    "${CMAKE_CURRENT_SOURCE_DIR}/Utility/*.h")
file(GLOB HEADERS_VISUALIZER "${CMAKE_CURRENT_SOURCE_DIR}/Visualizer/*.h")

install(FILES ${HEADERS_ROOT}       DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Visualization)
install(FILES ${HEADERS_SHADER}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Visualization/Shader)
install(FILES ${HEADERS_UTILITY}    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Visualization/Utility)
install(FILES ${HEADERS_VISUALIZER} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Visualization/Visualizer)
