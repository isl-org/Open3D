# On Ubuntu 20.04, get the latest CMake from https://apt.kitware.com/.
cmake_minimum_required(VERSION 3.24)

# find_package(Open3D) can reset CMP0091 via old dependency cmake_policy() calls.
# With CMP0091 not NEW, CMAKE_MSVC_RUNTIME_LIBRARY is ignored and Draw defaults
# to /MT, causing LNK2038 against /MD Open3D. Set the policy default before
# project() (same idea as 3rdparty/find_dependencies.cmake).
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)

project(Open3DCMakeFindPackage LANGUAGES C CXX)

# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)
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

# Match Open3D's /MT or /MD (CMP0091 above); explicit /M flags are a fallback if
# a dependency resets CMP0091 and MSVC_RUNTIME_LIBRARY is ignored.
if(MSVC)
    set_target_properties(Draw PROPERTIES
        MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
    if(STATIC_WINDOWS_RUNTIME)
        target_compile_options(Draw PRIVATE $<IF:$<CONFIG:Debug>,/MTd,/MT>)
    else()
        target_compile_options(Draw PRIVATE $<IF:$<CONFIG:Debug>,/MDd,/MD>)
    endif()
endif()

# On Windows if BUILD_SHARED_LIBS is enabled, copy .dll files to the executable directory
if(WIN32)
    get_target_property(open3d_type Open3D::Open3D TYPE)
    get_filename_component(OPEN3D_INSTALL_PREFIX "${Open3D_DIR}/.." ABSOLUTE)
    if(open3d_type STREQUAL "SHARED_LIBRARY")
        set(copy_dlls "${OPEN3D_INSTALL_PREFIX}/bin/tbb12$<$<CONFIG:Debug>:_debug>.dll" 
                      "${OPEN3D_INSTALL_PREFIX}/bin/Open3D.dll")
    else() 
        set(copy_dlls "${OPEN3D_INSTALL_PREFIX}/bin/tbb12$<$<CONFIG:Debug>:_debug>.dll") 
    endif()
    # For CUDA builds, we must also copy the CUDA DLLs. Alternately, add
    # $Env:CUDA_PATH/bin to $Env:PATH to avoid this copy.
    add_custom_command(TARGET Draw POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy ${copy_dlls}
                        $<TARGET_FILE_DIR:Draw>
                        COMMENT "Copying Open3D DLLs to $<TARGET_FILE_DIR:Draw>")
endif()
if (MSVC)
    add_compile_options(/utf-8)  # fmt 12+ requires /utf-8 with MSVC
endif()
