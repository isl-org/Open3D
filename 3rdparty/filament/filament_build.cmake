include(ExternalProject)

set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament-binaries")

set(FILAMENT_GIT_REPOSITORY "https://github.com/intel-isl/filament.git")
set(FILAMENT_GIT_TAG "13ad8e25289cb173a4f8e71e85cb4e0d026eacdc")

set(filament_LIBRARIES
    filameshio
    filament
    filamat_lite
    filamat
    filaflat
    filabridge
    geometry
    backend
    bluegl
    ibl
    image
    meshoptimizer
    smol-v
    utils
)

set(lib_dir lib/x86_64)
if(MSVC)
    if (STATIC_WINDOWS_RUNTIME)
        string(APPEND lib_dir /mt)
    else()
        string(APPEND lib_dir /md)
    endif()
endif()

set(lib_byproducts ${filament_LIBRARIES})
list(TRANSFORM lib_byproducts PREPEND ${FILAMENT_ROOT}/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
if(WIN32)
    set(lib_byproducts_debug ${filament_LIBRARIES})
    list(TRANSFORM lib_byproducts_debug PREPEND ${FILAMENT_ROOT}/${lib_dir}d/${CMAKE_STATIC_LIBRARY_PREFIX})
    list(TRANSFORM lib_byproducts_debug APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
    list(APPEND lib_byproducts ${lib_byproducts_debug})
endif()

ExternalProject_Add(
    ext_filament
    PREFIX filament
    GIT_REPOSITORY ${FILAMENT_GIT_REPOSITORY}
    GIT_TAG ${FILAMENT_GIT_TAG}
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCCACHE_PROGRAM=OFF  # Enables ccache, "launch-cxx" is not working.
        -DFILAMENT_ENABLE_JAVA=OFF
        -DCMAKE_C_COMPILER=${FILAMENT_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${FILAMENT_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CXX_FLAGS="-fno-builtin"  # Issue Open3D#1909, filament#2146
        -DCMAKE_INSTALL_PREFIX=${FILAMENT_ROOT}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DUSE_STATIC_CRT=${STATIC_WINDOWS_RUNTIME}
        -DUSE_STATIC_LIBCXX=ON
        -DFILAMENT_SUPPORTS_VULKAN=OFF
        -DFILAMENT_SKIP_SAMPLES=ON
        -DFILAMENT_OPENGL_HANDLE_ARENA_SIZE_IN_MB=20 # to support many small entities
    BUILD_BYPRODUCTS ${lib_byproducts}
)
