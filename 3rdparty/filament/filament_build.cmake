include(ExternalProject)

set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament-binaries")

# Handle build type for single and multi-config generators.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(FILAMENT_BUILD_TYPE ${CMAKE_BUILD_TYPE})
if(NOT is_multi_config)
    # Do not mix debug/release CRT on Windows.
    if (NOT MSVC)
        set(FILAMENT_BUILD_TYPE "Release")
    endif()
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

# Locate byproducts
set(lib_dir lib)
if(APPLE)
    if(APPLE_AARCH64)
        set(lib_dir lib/arm64)
    else()
        set(lib_dir lib/x86_64)
    endif()
endif()

set(lib_byproducts ${filament_LIBRARIES})
list(TRANSFORM lib_byproducts PREPEND ${FILAMENT_ROOT}/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})

set(filament_cxx_flags "${CMAKE_CXX_FLAGS} -Wno-deprecated" "-Wno-pass-failed=transform-warning" "-Wno-error=nonnull")
if(NOT WIN32)
    # Issue Open3D#1909, filament#2146
    set(filament_cxx_flags "${filament_cxx_flags} -fno-builtin")
    # Ensure we use the correct LLVM version's shared libc++ (not static libc++ from older LLVM)
    # This prevents linker errors with static libc++ that wasn't built with -fPIC
    if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        # Find the Clang version to locate the correct libc++
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} --version
            OUTPUT_VARIABLE clang_version_output
            ERROR_QUIET
        )
        if(clang_version_output MATCHES "clang version ([0-9]+)")
            set(CLANG_VER "${CMAKE_MATCH_1}")
            set(LLVM_LIBDIR "/usr/lib/llvm-${CLANG_VER}/lib")
            set(LLVM_INCDIR "/usr/lib/llvm-${CLANG_VER}/include")
            # Use LLVM 14's shared libc++ if available (it's built with -fPIC)
            if(EXISTS "${LLVM_LIBDIR}/libc++.so")
                set(filament_cxx_flags "${filament_cxx_flags} -L${LLVM_LIBDIR}")
                set(filament_ld_flags "-Wl,-rpath,${LLVM_LIBDIR}")
                # Force-include cstdint to fix missing uint32_t in Filament headers
                # This is a workaround for Filament's TIFFExport.h missing #include <cstdint>
                if(EXISTS "${LLVM_INCDIR}/c++/v1/cstdint")
                    set(filament_cxx_flags "${filament_cxx_flags} -include ${LLVM_INCDIR}/c++/v1/cstdint")
                endif()
                message(STATUS "Filament: Using LLVM ${CLANG_VER} shared libc++ from ${LLVM_LIBDIR}")
            endif()
        endif()
    endif()
endif()

ExternalProject_Add(
    ext_filament
    PREFIX filament
    URL https://github.com/google/filament/archive/refs/tags/v1.54.0.tar.gz
    URL_HASH SHA256=f4cb4eb81e3a5d66a9612ac131d16183e118b694f4f34c051506c523a8389e8d
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/filament"
    UPDATE_COMMAND ""
    PATCH_COMMAND
        bash -c "sed -i '/^#include <fstream>/a #include <cstdint>' libs/viewer/src/TIFFExport.h || true"
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS}
        -DCMAKE_BUILD_TYPE=${FILAMENT_BUILD_TYPE}
        -DCCACHE_PROGRAM=OFF  # Enables ccache, "launch-cxx" is not working.
        -DFILAMENT_ENABLE_JAVA=OFF
        -DCMAKE_C_COMPILER=${FILAMENT_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${FILAMENT_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CXX_FLAGS:STRING=${filament_cxx_flags}
        -DCMAKE_EXE_LINKER_FLAGS:STRING=${filament_ld_flags}
        -DCMAKE_SHARED_LINKER_FLAGS:STRING=${filament_ld_flags}
        -DCMAKE_INSTALL_PREFIX=${FILAMENT_ROOT}
        -DUSE_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
        -DUSE_STATIC_LIBCXX=ON
        -DFILAMENT_SKIP_SDL2=ON
        -DFILAMENT_SKIP_SAMPLES=ON
        -DFILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB=20 # to support many small entities
        -DSPIRV_WERROR=OFF
        BUILD_BYPRODUCTS ${lib_byproducts}
)