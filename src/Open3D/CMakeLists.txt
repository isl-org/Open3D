# Configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/src/Open3D/Open3D.h.in"
               "${PROJECT_SOURCE_DIR}/src/Open3D/Open3D.h")
configure_file("${PROJECT_SOURCE_DIR}/src/Open3D/Open3DConfig.h.in"
               "${PROJECT_SOURCE_DIR}/src/Open3D/Open3DConfig.h")

# Subdirectories
add_subdirectory(Camera)
add_subdirectory(ColorMap)
add_subdirectory(Core)
add_subdirectory(Geometry)
add_subdirectory(Integration)
add_subdirectory(Odometry)
add_subdirectory(Registration)
add_subdirectory(Utility)
add_subdirectory(IO)
add_subdirectory(Visualization)

# Source group for Visual Studio
add_source_group(Camera)
add_source_group(ColorMap)
add_source_group(Core)
add_source_group(Geometry)
add_source_group(Integration)
add_source_group(Odometry)
add_source_group(Registration)
add_source_group(Utility)
add_source_group(IO)
add_source_group(Visualization)

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
add_library(${PROJECT_NAME}
    Open3DConfig.h
    Open3DConfig.cpp
    $<TARGET_OBJECTS:Camera>
    $<TARGET_OBJECTS:ColorMap>
    $<TARGET_OBJECTS:Core>
    $<TARGET_OBJECTS:Geometry>
    $<TARGET_OBJECTS:Integration>
    $<TARGET_OBJECTS:Odometry>
    $<TARGET_OBJECTS:Registration>
    $<TARGET_OBJECTS:Utility>
    $<TARGET_OBJECTS:IO>
    $<TARGET_OBJECTS:Visualization>
)
open3d_show_and_abort_on_warning(${PROJECT_NAME})
open3d_set_global_properties(${PROJECT_NAME})
open3d_link_3rdparty_libraries(${PROJECT_NAME})

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
if (NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
                   IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
