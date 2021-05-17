# some black magic to automatically create a header file from shader files
file(GLOB SHADER_FILES "shader/glsl/*.glsl")

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/shader/Shader.h
    COMMAND EncodeShader Shader.h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/shader/
    DEPENDS ${SHADER_FILES} EncodeShader
)

foreach(file ${SHADER_FILES})
    add_custom_command(
        OUTPUT ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/shader/Shader.h
        COMMAND EncodeShader Shader.h ${file}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/shader/
        DEPENDS ${SHADER_FILES} EncodeShader
        APPEND
    )
endforeach(file)

add_custom_target(ShaderFileTarget ALL
                  DEPENDS ${PROJECT_SOURCE_DIR}/cpp/open3d/visualization/shader/Shader.h)
set_target_properties(ShaderFileTarget PROPERTIES FOLDER "Custom")

# build
set(VISUALIZATION_ALL_HEADER_FILES "")
set(VISUALIZATION_ALL_SOURCE_FILES "")
macro(AddSourceDir DIR)
    file(GLOB_RECURSE HEADER_FILES "${DIR}/*.h")
    file(GLOB_RECURSE SOURCE_FILES "${DIR}/*.cpp")
    list(APPEND VISUALIZATION_ALL_HEADER_FILES ${HEADER_FILES})
    list(APPEND VISUALIZATION_ALL_SOURCE_FILES ${SOURCE_FILES})
endmacro(AddSourceDir)
AddSourceDir("shader")
AddSourceDir("utility")
AddSourceDir("visualizer")

if (BUILD_GUI)
    AddSourceDir("rendering")
else ()
    list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*GuiWidgets.cpp)
    list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*GuiWidgets.h)
    list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*/Draw.h)
    list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*/Draw.cpp)
    list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*O3D.*.h)
    list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*O3D.*.cpp)
endif()

if ( NOT BUILD_RPC_INTERFACE )
        list(FILTER VISUALIZATION_ALL_SOURCE_FILES EXCLUDE REGEX .*Receiver.cpp)
endif()

# create object library
add_library(visualization OBJECT
            ${SHADER_FILES}
            ${VISUALIZATION_ALL_HEADER_FILES}
            ${VISUALIZATION_ALL_SOURCE_FILES})

source_group("Source Files\\Shader\\GLSL" FILES ${SHADER_FILES})

add_dependencies(visualization ShaderFileTarget)

open3d_show_and_abort_on_warning(visualization)
open3d_set_global_properties(visualization)
open3d_set_open3d_lib_properties(visualization)
open3d_link_3rdparty_libraries(visualization)

# export GUI_RESOURCE_DIR to parent scope
if (BUILD_GUI)
    set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
endif()
