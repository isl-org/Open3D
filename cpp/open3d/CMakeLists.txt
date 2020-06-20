# Configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/cpp/open3d/Open3D.h.in"
               "${PROJECT_SOURCE_DIR}/cpp/open3d/Open3D.h")
configure_file("${PROJECT_SOURCE_DIR}/cpp/open3d/Open3DConfig.h.in"
               "${PROJECT_SOURCE_DIR}/cpp/open3d/Open3DConfig.h")

# Subdirectories
add_subdirectory(camera)
add_subdirectory(core)
add_subdirectory(geometry)
add_subdirectory(io)
add_subdirectory(ml)
add_subdirectory(pipelines)
add_subdirectory(utility)
add_subdirectory(visualization)
if (ENABLE_GUI)
    add_subdirectory(visualization/gui)
    set(GUI_OBJECTS "$<TARGET_OBJECTS:GUI>")
endif()

# Source group for Visual Studio
add_source_group(camera)
add_source_group(core)
add_source_group(geometry)
add_source_group(io)
add_source_group(ml)
add_source_group(pipelines)
add_source_group(utility)
add_source_group(visualization)

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
add_library(${CMAKE_PROJECT_NAME}
    Open3DConfig.cpp
    $<TARGET_OBJECTS:camera>
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:geometry>
    $<TARGET_OBJECTS:io>
    $<TARGET_OBJECTS:pipelines>
    $<TARGET_OBJECTS:utility>
    $<TARGET_OBJECTS:visualization>
    ${GUI_OBJECTS}
)
open3d_show_and_abort_on_warning(${PROJECT_NAME})
open3d_set_global_properties(${PROJECT_NAME})
open3d_link_3rdparty_libraries(${PROJECT_NAME})
if(X11_TARGET)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${X11_TARGET})
endif()
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

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
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
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
