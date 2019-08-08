# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "Shader/GLSL/*.glsl")

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/src/Open3D/Visualization/Shader/Shader.h
    COMMAND EncodeShader Shader.h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Open3D/Visualization/Shader/
    DEPENDS ${SHADER_FILES} EncodeShader
)

foreach(file ${SHADER_FILES})
    add_custom_command(
        OUTPUT ${PROJECT_SOURCE_DIR}/src/Open3D/Visualization/Shader/Shader.h
        COMMAND EncodeShader Shader.h ${file}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Open3D/Visualization/Shader/
        DEPENDS ${SHADER_FILES} EncodeShader
        APPEND
    )
endforeach(file)

add_custom_target(ShaderFileTarget ALL
                  DEPENDS ${PROJECT_SOURCE_DIR}/src/Open3D/Visualization/Shader/Shader.h)
set_target_properties(ShaderFileTarget PROPERTIES FOLDER "Custom")

# build
file(GLOB_RECURSE VISUALIZATION_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE VISUALIZATION_ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Visualization OBJECT
            ${SHADER_FILES}
            ${VISUALIZATION_ALL_HEADER_FILES}
            ${VISUALIZATION_ALL_SOURCE_FILES})

source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})

add_dependencies(Visualization ShaderFileTarget)

ShowAndAbortOnWarning(Visualization)

# Enforce 3rd party dependencies
add_dependencies(Visualization build_all_3rd_party_libs)
