# open3d_set_open3d_lib_properties() sets properties for the Open3D lib itself.
# This should be used for all object libraries that make up the Open3D lib.
#
# In comparison, open3d_set_global_properties() sets properties for the Open3D
# lib, and targets that links to the Open3D lib, e.g pybind, unit tests, etc.
function(open3d_set_open3d_lib_properties target)
    cmake_parse_arguments(arg "HIDDEN" "" "" ${ARGN})
    if(NOT BUILD_SHARED_LIBS)
        target_compile_definitions(${target} PUBLIC OPEN3D_STATIC)
    endif()
    if (arg_HIDDEN)
        set_target_properties(${target} PROPERTIES
                        CXX_VISIBILITY_PRESET hidden
                        VISIBILITY_INLINES_HIDDEN ON
                        )
    else ()
        target_compile_definitions(${target} PRIVATE OPEN3D_ENABLE_DLL_EXPORTS)
    endif()

endfunction()

# https://stackoverflow.com/a/65177331
# file(REAL_PATH "${CMAKE_INSTALL_PREFIX}" prefix)
function(auto_pc TARGET)
  file(CONFIGURE OUTPUT "pc.${TARGET}/CMakeLists.txt"
       CONTENT [[
cmake_minimum_required(VERSION 3.20)
project(pc_@TARGET@)

find_package(@TARGET@ REQUIRED CONFIG PATHS @CMAKE_INSTALL_PREFIX@ NO_DEFAULT_PATH)

file(GENERATE OUTPUT lib@TARGET@.pc
     CONTENT [=[
prefix=@CMAKE_INSTALL_PREFIX@
libdir=${prefix}/lib
includedir=${prefix}/include/

Name: lib@TARGET@
Description: @PROJECT_DESCRIPTION@
URL: @PROJECT_HOMEPAGE_URL@
Version: @PROJECT_VERSION@
Cflags: -I$<JOIN:$<TARGET_PROPERTY:INTERFACE_INCLUDE_DIRECTORIES>, -I> -D$<JOIN:$<TARGET_PROPERTY:INTERFACE_COMPILE_DEFINITIONS>, -D>
Libs: -L$<TARGET_FILE_DIR:Open3D::@TARGET@> -l@TARGET@
Libs.private: $<JOIN:$<TARGET_PROPERTY:INTERFACE_LINK_OPTIONS>, >
$<JOIN:$<TARGET_PROPERTY:INTERFACE_LINK_LIBRARIES>, >
]=]   TARGET "Open3D::@TARGET@")
]] @ONLY NEWLINE_STYLE LF)

  file(CONFIGURE OUTPUT "pc.${TARGET}/post-install.cmake"
       CONTENT [[
set(proj "@CMAKE_CURRENT_BINARY_DIR@/pc.@TARGET@")
execute_process(COMMAND "@CMAKE_COMMAND@" -S "${proj}" -B "${proj}/build")
file(INSTALL "${proj}/build/lib@TARGET@.pc" DESTINATION "@CMAKE_INSTALL_PREFIX@/@Open3D_INSTALL_LIB_DIR@/pkgconfig")
]] @ONLY NEWLINE_STYLE LF)

  install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/pc.${TARGET}/post-install.cmake")
endfunction()

# Configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/cpp/open3d/Open3D.h.in"
               "${PROJECT_SOURCE_DIR}/cpp/open3d/Open3D.h")
configure_file("${PROJECT_SOURCE_DIR}/cpp/open3d/Open3DConfig.h.in"
               "${PROJECT_SOURCE_DIR}/cpp/open3d/Open3DConfig.h")


add_library(Open3D)

add_subdirectory(camera)
add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(ml)
add_subdirectory(pipelines)
add_subdirectory(t/geometry)
add_subdirectory(t/io)
add_subdirectory(t/pipelines)
add_subdirectory(utility)
add_subdirectory(visualization)

if (BUILD_GUI)
    add_subdirectory(visualization/gui)
endif()

if (BUILD_WEBRTC)
    add_subdirectory(visualization/webrtc_server)
endif()

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
target_sources(Open3D PRIVATE
    Open3DConfig.cpp
)

open3d_ispc_target_sources_TARGET_OBJECTS(Open3D PRIVATE
    camera
    core
    data
    geometry
    tgeometry
    tgeometry_kernel
    io
    tio
    ml_contrib
    pipelines
    tpipelines
    tpipelines_kernel
    utility
    visualization
)

if (BUILD_GUI)
    open3d_ispc_target_sources_TARGET_OBJECTS(Open3D PRIVATE
        GUI
    )
endif()

if (BUILD_WEBRTC)
    open3d_ispc_target_sources_TARGET_OBJECTS(Open3D PRIVATE
        webrtc_server
    )
endif()

# Source group for Visual Studio
add_source_group(camera)
add_source_group(core)
add_source_group(data)
add_source_group(geometry)
add_source_group(tgeometry)
add_source_group(io)
add_source_group(tio)
add_source_group(ml)
add_source_group(pipelines)
add_source_group(tpipelines)
add_source_group(utility)
add_source_group(visualization)

open3d_show_and_abort_on_warning(Open3D)
open3d_set_global_properties(Open3D)
open3d_set_open3d_lib_properties(Open3D)
open3d_link_3rdparty_libraries(Open3D)

# If we are building a STATIC_LIBRARY, hide symbols coming from 3rd party static
# libraries that are not hidden during compilation. Don't propagate beyond
# direct consumers of libOpen3D.a
target_link_options(Open3D INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:Open3D,TYPE>,STATIC_LIBRARY>:$<LINK_ONLY:${OPEN3D_HIDDEN_3RDPARTY_LINK_OPTIONS}>>)

add_library(Open3D::Open3D ALIAS Open3D)

include(CMakePackageConfigHelpers)

# find_package Open3D
configure_package_config_file(Open3DConfig.cmake.in
                              "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Open3DConfig.cmake"
                              INSTALL_DESTINATION ${Open3D_INSTALL_CMAKE_DIR}
                              PATH_VARS Open3D_INSTALL_INCLUDE_DIR
                              NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# find_package Open3D Version
write_basic_package_version_file("${PROJECT_BINARY_DIR}/Open3DConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY ExactVersion)

# Installation
install(TARGETS Open3D EXPORT Open3DTargets
        RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cuh"
)

# Install the Open3DConfig.cmake and Open3DConfigVersion.cmake
install(FILES
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Open3DConfig.cmake"
        "${PROJECT_BINARY_DIR}/Open3DConfigVersion.cmake"
        DESTINATION "${Open3D_INSTALL_CMAKE_DIR}" COMPONENT dev)

auto_pc(Open3D)

# uninstall target
if(NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
                   IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# Export GUI_RESOURCE_FILES to parent CMake context (cpp/open3d/)
set(GUI_RESOURCE_FILES ${GUI_RESOURCE_FILES} PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)
