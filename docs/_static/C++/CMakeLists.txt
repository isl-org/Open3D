cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 14)

# The options shall be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(BUILD_SHARED_LIBS "Whether Open3D was build as shared library" OFF)

project(TestVisualizer LANGUAGES C CXX)

# Find installed Open3D, which exports Open3D::Open3D
if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
endif()
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()

add_executable(TestVisualizer TestVisualizer.cpp)
target_link_libraries(TestVisualizer Open3D::Open3D)

# On Windows, when BUILD_SHARED_LIBS, copy .dll to the executable directory
if(WIN32)
    set(DIR_SUFFIX $<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Release>)
    if(BUILD_SHARED_LIBS)
        message("Will copy Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/${DIR_SUFFIX}")
        add_custom_command(TARGET TestVisualizer POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy
                                ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                                ${CMAKE_CURRENT_BINARY_DIR}/${DIR_SUFFIX})
    endif()
    if(ENABLE_GUI)
        message("Will copy SDL2.dll to ${CMAKE_CURRENT_BINARY_DIR}/${DIR_SUFFIX}")
        add_custom_command(TARGET TestVisualizer POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy
                                ${CMAKE_INSTALL_PREFIX}/lib/SDL2.dll
                                ${CMAKE_CURRENT_BINARY_DIR}/${DIR_SUFFIX})
    endif()
endif()
