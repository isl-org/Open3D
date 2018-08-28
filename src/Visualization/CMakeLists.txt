# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "Shader/GLSL/*.glsl")

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/src/Visualization/Shader/Shader.h
    COMMAND EncodeShader Shader.h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Visualization/Shader/
    DEPENDS ${SHADER_FILES} EncodeShader
)

foreach(file ${SHADER_FILES})
    add_custom_command(
        OUTPUT ${PROJECT_SOURCE_DIR}/src/Visualization/Shader/Shader.h
        COMMAND EncodeShader Shader.h ${file}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Visualization/Shader/
        DEPENDS ${SHADER_FILES} EncodeShader
        APPEND
    )
endforeach(file)

add_custom_target(ShaderFileTarget ALL
                  DEPENDS ${PROJECT_SOURCE_DIR}/src/Visualization/Shader/Shader.h)
set_target_properties(ShaderFileTarget PROPERTIES FOLDER "Custom")

# build
file(GLOB_RECURSE VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE VISUALIZATION_ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Visualization OBJECT
            ${SHADER_FILES}
            ${VISUALIZATION_ALL_HEADER_FILES}
            ${VISUALIZATION_ALL_SOURCE_FILES})

ADD_SOURCE_GROUP(Shader)
source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})
ADD_SOURCE_GROUP(Utility)
ADD_SOURCE_GROUP(Visualizer)

add_dependencies(Visualization ShaderFileTarget)

Directories(${CMAKE_CURRENT_SOURCE_DIR} VISUALIZATION_INCLUDE_DIRS)
set(VISUALIZATION_INCLUDE_DIRS ${VISUALIZATION_INCLUDE_DIRS} PARENT_SCOPE)

target_include_directories(Visualization PUBLIC ${VISUALIZATION_INCLUDE_DIRS})

#install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
        PATTERN     "Shader/GLSL" EXCLUDE
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)
