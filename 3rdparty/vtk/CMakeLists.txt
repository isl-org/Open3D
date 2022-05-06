# This file is used for creating the precompiled VTK archives that will be
# downloaded when building Open3D. This will make use of vtk_build.cmake.
# See .github/workflows/vtk_packages.yml for more information.
cmake_minimum_required(VERSION 3.19.2)
project(VTKArchiveBuild)

option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" OFF)

# OS specific settings
if(WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explictly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

if (APPLE)
set (CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING
    "Minimum OS X deployment version" FORCE)
endif()

if(UNIX AND NOT APPLE)
    execute_process(COMMAND uname -p
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PROCESSOR_ARCH STREQUAL "aarch64")
        set(LINUX_AARCH64 TRUE)
    endif()
endif()
if(APPLE)
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PROCESSOR_ARCH STREQUAL "arm64")
        set(APPLE_AARCH64 TRUE)
    endif()
endif()

# copy of var definitions from find_dependencies.cmake

# CMake arguments for configuring ExternalProjects. Use the second _hidden
# version by default.
set(ExternalProject_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
    -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_CUDA_COMPILER_LAUNCHER=${CMAKE_CUDA_COMPILER_LAUNCHER}
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
    # Always build 3rd party code in Release mode. Ignored by multi-config
    # generators (XCode, MSVC). MSVC needs matching config anyway.
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW
    -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)
# Keep 3rd party symbols hidden from Open3D user code. Do not use if 3rd party
# libraries throw exceptions that escape Open3D.
set(ExternalProject_CMAKE_ARGS_hidden
    ${ExternalProject_CMAKE_ARGS}
    # Apply LANG_VISIBILITY_PRESET to static libraries and archives as well
    -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_CUDA_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
)

set(OPEN3D_THIRD_PARTY_DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/../../3rdparty_downloads")
set(Open3D_INSTALL_LIB_DIR lib)

set(BUILD_VTK_FROM_SOURCE ON)
include(vtk_build.cmake)

if(LINUX_AARCH64)
    set(ARCHIVE_SUFFIX "_linux_aarch64")
elseif(APPLE_AARCH64)
    set(ARCHIVE_SUFFIX "_macos_${CMAKE_OSX_DEPLOYMENT_TARGET}_arm64")
elseif(APPLE)
    set(ARCHIVE_SUFFIX "_macos_${CMAKE_OSX_DEPLOYMENT_TARGET}")
elseif(UNIX)
    set(ARCHIVE_SUFFIX "_linux_x86_64")
elseif(WIN32)
    if (STATIC_WINDOWS_RUNTIME)
        set(ARCHIVE_SUFFIX "_win_staticrt")
    else()
        set(ARCHIVE_SUFFIX "_win")
    endif()
else()
    message(FATAL "Unsupported platform")
endif()

set(ARCHIVE_NAME "vtk_${VTK_VERSION}${ARCHIVE_SUFFIX}.tar.gz")

add_custom_target(vtk_archive ALL
    COMMENT "Creating tar archive"
    COMMAND ${CMAKE_COMMAND} -E tar -cvzf ${ARCHIVE_NAME} vtk/include vtk/lib vtk/share
    COMMAND ${CMAKE_COMMAND} -E sha256sum ${ARCHIVE_NAME}
    DEPENDS ext_vtk
)
