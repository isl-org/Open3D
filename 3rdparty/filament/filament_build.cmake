include(ExternalProject)

set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament-binaries")

# Handle build type for single and multi-config generators.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    # Select the optimized configuration when the parent build requests Debug.
    set(FILAMENT_BUILD_TYPE "RelWithDebInfo")
    set(FILAMENT_BUILD_CONFIG
        "$<IF:$<CONFIG:Debug>,RelWithDebInfo,$<CONFIG>>")
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR
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
        vkshaders
)

set(FILAMENT_VER "v1.54.0")
set(FILAMENT_VER_HASH "f4cb4eb81e3a5d66a9612ac131d16183e118b694f4f34c051506c523a8389e8d")

# Locate byproducts
set(lib_dir lib)
if(APPLE)
    set(FILAMENT_VER "v1.57.2")    # Metal shared texture support for 3DGS
    set(FILAMENT_VER_HASH "")
    if(APPLE_AARCH64)
        set(lib_dir lib/arm64)
    else()
        set(lib_dir lib/x86_64)
    endif()
endif()

set(lib_byproducts ${filament_LIBRARIES})
list(TRANSFORM lib_byproducts PREPEND ${FILAMENT_ROOT}/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})

set(filament_cxx_flags "${CMAKE_CXX_FLAGS}")
if(NOT MSVC)
    set(filament_cxx_flags "${filament_cxx_flags} -Wno-deprecated"
        "-Wno-pass-failed=transform-warning" "-Wno-error=nonnull")
endif()
if(NOT WIN32)
    # Issue Open3D#1909, filament#2146
    set(filament_cxx_flags "${filament_cxx_flags} -fno-builtin")
endif()

# Clang on Linux needs the GCC libstdc++ library path for linking.
# When building Filament from source with clang while the rest of
# Open3D uses GCC, the GCC library directory must be explicitly added.
set(filament_linker_flags "")
if(UNIX AND NOT APPLE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-search-dirs
        OUTPUT_VARIABLE _gcc_search_dirs
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_gcc_search_dirs MATCHES "libraries: *=([^\n]+)")
        string(STRIP "${CMAKE_MATCH_1}" _gcc_lib_path)
        # Pick the first path that contains libstdc++.
        string(REPLACE ":" ";" _gcc_lib_list "${_gcc_lib_path}")
        foreach(_dir ${_gcc_lib_list})
            if(EXISTS "${_dir}/libstdc++.so")
                set(filament_linker_flags "-L${_dir}")
                break()
            endif()
        endforeach()
    endif()
    # Fallback: try common locations.
    if(NOT filament_linker_flags)
        foreach(_dir /usr/lib/gcc/x86_64-linux-gnu/11
                     /usr/lib/gcc/x86_64-linux-gnu/12
                     /usr/lib/gcc/x86_64-linux-gnu/13
                     /usr/lib/x86_64-linux-gnu)
            if(EXISTS "${_dir}/libstdc++.so")
                set(filament_linker_flags "-L${_dir}")
                break()
            endif()
        endforeach()
    endif()
    if(filament_linker_flags)
        message(STATUS "Filament: using linker flags ${filament_linker_flags}")
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
    PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH_FILE=${Open3D_3RDPARTY_DIR}/filament/patches/0001-importTextureR.patch -DSOURCE_DIR=<SOURCE_DIR> -P ${Open3D_3RDPARTY_DIR}/librealsense/apply_patch.cmake
    # 0002: Backports upstream Vulkan swapchain acquire retry/recovery on window resize.
    COMMAND ${CMAKE_COMMAND} -DPATCH_FILE=${Open3D_3RDPARTY_DIR}/filament/patches/0002-handle-vulkan-swapchain-acquire.patch -DSOURCE_DIR=<SOURCE_DIR> -P ${Open3D_3RDPARTY_DIR}/librealsense/apply_patch.cmake
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS}
        -DCMAKE_BUILD_TYPE=${FILAMENT_BUILD_TYPE}
        -DCCACHE_PROGRAM=OFF  # Enables ccache, "launch-cxx" is not working.
        -DFILAMENT_ENABLE_JAVA=OFF
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
