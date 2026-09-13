# TODO: Cleanup libc++ handling for filament in Linux.
# Filament needs libc++ - will not compile with libstdc++ instead in Linux.
# llvm libc++ < 11 (Ubuntu 20.04 or older): libc++ and libc++abi needed. We can
# package libc++, libc++abi with Open3D and things work correctly.
# llvm libc++ >=1 : Also needs llvm libunwind - this conflicts with system libunwind
# and leads to crashes when exceptions are used. No way to isolate llvm libc++,
# libc++abi and libunwind away from the rest of libstdc++ based Open3D.
# From Open3D v0.20+, we instead build llvm libc++stdabi.a: static libc++ linked
# against libstdc++ for low level ABi and unwind. This is packaged with
# filament. No separate llvm libraries are needed, and the system libunwind and
# libstdc++ works correctly.

include(ExternalProject)

set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament-binaries")

# Handle build type for single and multi-config generators.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    if(MSVC)
        # MSVC Debug uses a distinct CRT and STL iterator ABI.
        set(FILAMENT_BUILD_TYPE "Debug")
        set(FILAMENT_BUILD_CONFIG "$<CONFIG>")
    else()
        # Keep Filament optimized when the parent build requests Debug.
        set(FILAMENT_BUILD_TYPE "RelWithDebInfo")
        set(FILAMENT_BUILD_CONFIG
            "$<IF:$<CONFIG:Debug>,RelWithDebInfo,$<CONFIG>>")
    endif()
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND MSVC)
        set(FILAMENT_BUILD_TYPE "Debug")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug" OR
           CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        # Keep Filament optimized when Open3D is built with debug information.
        set(FILAMENT_BUILD_TYPE "RelWithDebInfo")
    else()
        # Do not mix debug/release CRT on Windows.
        set(FILAMENT_BUILD_TYPE "Release")
    endif()
    set(FILAMENT_BUILD_CONFIG ${FILAMENT_BUILD_TYPE})
endif()

set(filament_LIBRARIES
        filameshio
        filament
        filaflat
        filabridge
        geometry
        backend
        bluegl
        bluevk
        ibl
        image
        ktxreader
        meshoptimizer
        smol-v
        utils
        zstd
)

set(FILAMENT_VER "v1.76.0")
set(FILAMENT_VER_HASH "8cee7aa7aee72d2a62b5a1b3871d9f6d9e276e982fc4917960ac1da5c5b425dc")

# Locate byproducts
set(lib_dir lib)
if(APPLE_AARCH64)
    set(lib_dir lib/arm64)
else()
    set(lib_dir lib/x86_64)
endif()
list(APPEND filament_LIBRARIES shaders)

set(lib_byproducts ${filament_LIBRARIES})
list(TRANSFORM lib_byproducts PREPEND ${FILAMENT_ROOT}/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})

set(filament_cxx_flags "${CMAKE_CXX_FLAGS}")
if(NOT MSVC)
    set(filament_cxx_flags "${filament_cxx_flags} -Wno-deprecated"
        "-Wno-error=nonnull")
endif()
if(NOT WIN32)
    # Issue Open3D#1909, filament#2146
    set(filament_cxx_flags "${filament_cxx_flags} -fno-builtin")
endif()

# Clang needs the system GCC libstdc++ directory when Filament is built from
# source on supported Debian-based Linux distributions.
set(filament_linker_flags "")
if(UNIX AND NOT APPLE)
    execute_process(COMMAND g++ -print-file-name=libstdc++.so
        OUTPUT_VARIABLE filament_libstdcxx
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(EXISTS "${filament_libstdcxx}")
        get_filename_component(filament_libstdcxx_dir
                               "${filament_libstdcxx}" DIRECTORY)
        set(filament_linker_flags "-L${filament_libstdcxx_dir}")
        message(STATUS "Filament: using linker flags ${filament_linker_flags}")
    else()
        message(FATAL_ERROR "Could not locate the system libstdc++.so")
    endif()
endif()

if(MSVC)
    set(filament_build_command
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${FILAMENT_BUILD_CONFIG})
endif()

ExternalProject_Add(
    ext_filament
    PREFIX filament
    URL https://github.com/google/filament/archive/refs/tags/${FILAMENT_VER}.tar.gz
    URL_HASH SHA256=${FILAMENT_VER_HASH}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/filament"
    # 0001: Implements VulkanDriver::importTextureR for zero-copy 3DGS texture sharing.
    PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH_FILE=${Open3D_3RDPARTY_DIR}/filament/patches/0001-importTextureR.patch -DSOURCE_DIR=<SOURCE_DIR> -P ${Open3D_SOURCE_DIR}/cmake/apply_patch.cmake
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS}
        -DCMAKE_BUILD_TYPE=${FILAMENT_BUILD_TYPE}
        -DCMAKE_CXX_STANDARD=20
        -DCMAKE_CXX_STANDARD_REQUIRED=ON
        -DCCACHE_PROGRAM=OFF  # Enables ccache, "launch-cxx" is not working.
        -DFILAMENT_ENABLE_JAVA=OFF
        -DFILAMENT_BUILD_TESTING=OFF
        -DFILAMENT_SUPPORTS_VULKAN=ON
        -DCMAKE_C_COMPILER=${FILAMENT_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${FILAMENT_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CXX_FLAGS:STRING=${filament_cxx_flags}
        -DCMAKE_EXE_LINKER_FLAGS:STRING=${filament_linker_flags}
        -DCMAKE_SHARED_LINKER_FLAGS:STRING=${filament_linker_flags}
        -DCMAKE_INSTALL_PREFIX=${FILAMENT_ROOT}
        -DUSE_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
        -DUSE_STATIC_LIBCXX=ON
        -DFILAMENT_SKIP_SDL2=ON
        -DFILAMENT_SKIP_SAMPLES=ON
        -DFILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB=20 # to support many small entities
        -DSPIRV_WERROR=OFF
        ${filament_build_command}
        BUILD_BYPRODUCTS ${lib_byproducts}
)
