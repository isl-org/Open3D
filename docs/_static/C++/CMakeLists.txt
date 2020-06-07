cmake_minimum_required(VERSION 3.12.0)
set (CMAKE_CXX_STANDARD 14)

if(POLICY CMP0091)
    # https://stackoverflow.com/a/56490614
    cmake_policy(SET CMP0091 NEW)
endif()

# The options shall be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)
if(STATIC_WINDOWS_RUNTIME)
    # This needs cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

project(TestVisualizer LANGUAGES C CXX)

# Find installed Open3D, which exports Open3D::Open3D
if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
endif()
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()

add_executable(TestVisualizer TestVisualizer.cpp)
target_link_libraries(TestVisualizer Open3D::Open3D)

# On Windows, when BUILD_SHARED_LIBS, copy .dll to the executable directory
if(WIN32)
    get_target_property(open3d_type Open3D::Open3D TYPE)
    if(open3d_type STREQUAL "SHARED_LIBRARY")
        message(STATUS "Will copy Open3D.dll to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
        add_custom_command(TARGET TestVisualizer POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy
                                ${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll
                                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    endif()
    if(ENABLE_GUI)
        if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib/SDL2.dll")
            message(STATUS "Will copy SDL2.dll to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
            add_custom_command(TARGET TestVisualizer POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy
                                    ${CMAKE_INSTALL_PREFIX}/lib/SDL2.dll
                                    ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
        else()
            message(WARNING "SDL2.dll not found in ${CMAKE_INSTALL_PREFIX}/lib")
        endif()
    endif()
endif()
