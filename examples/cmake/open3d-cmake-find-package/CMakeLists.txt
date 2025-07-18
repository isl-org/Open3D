# On Ubuntu 20.04, get the latest CMake from https://apt.kitware.com/.
cmake_minimum_required(VERSION 3.24)

project(Open3DCMakeFindPackage LANGUAGES C CXX)

# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
if(STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Find installed Open3D, which exports Open3D::Open3D
find_package(Open3D REQUIRED)

add_executable(Draw)
target_sources(Draw PRIVATE Draw.cpp)
target_link_libraries(Draw PRIVATE Open3D::Open3D)

# On Windows if BUILD_SHARED_LIBS is enabled, copy .dll files to the executable directory
if(WIN32)
    get_target_property(open3d_type Open3D::Open3D TYPE)
    if(open3d_type STREQUAL "SHARED_LIBRARY")
        set(copy_dlls "${CMAKE_INSTALL_PREFIX}/bin/tbb12$<$<CONFIG:Debug>:_debug>.dll" 
                      "${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll")
    else() 
        set(copy_dlls "${CMAKE_INSTALL_PREFIX}/bin/tbb12$<$<CONFIG:Debug>:_debug>.dll") 
    endif()
    # For CUDA builds, we must also copy the CUDA DLLs. Alternately, add
    # $Env:CUDA_PATH/bin to $Env:PATH to avoid this copy.
    add_custom_command(TARGET Draw POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy ${copy_dlls}
                        ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
                        COMMENT "Copying Open3D DLLs to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
endif()
