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
