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

# find_package(Open3D) transitively loads dependency configs, some of which call
# cmake_minimum_required()/cmake_policy() with an old version and can reset
# CMP0091 away from NEW in this scope. With CMP0091 not NEW, CMAKE_MSVC_RUNTIME_LIBRARY
# is silently ignored and MSVC defaults the target to the static runtime (/MT),
# which then fails to link (LNK2038) against a dynamic-runtime (/MD) Open3D.
# Re-assert the policy and the runtime library right before creating the target so
# the selection above is always honored, regardless of the runtime Open3D uses.
cmake_policy(SET CMP0091 NEW)

add_executable(Draw)
set_target_properties(Draw PROPERTIES MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
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
                        $<TARGET_FILE_DIR:Draw>
                        COMMENT "Copying Open3D DLLs to $<TARGET_FILE_DIR:Draw>")
endif()
if (MSVC)
    add_compile_options(/utf-8)  # fmt 12+ requires /utf-8 with MSVC
endif()
