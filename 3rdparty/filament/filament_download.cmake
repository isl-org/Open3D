include(FetchContent)

set(filament_LIBRARIES filameshio filament filamat_lite filaflat filabridge geometry backend bluegl bluevk ibl image meshoptimizer smol-v utils vkshaders)

if (FILAMENT_PRECOMPILED_ROOT)
    if (EXISTS "${FILAMENT_PRECOMPILED_ROOT}")
        set(FILAMENT_ROOT "${FILAMENT_PRECOMPILED_ROOT}")
    else()
        message(FATAL_ERROR "Filament binaries not found in ${FILAMENT_PRECOMPILED_ROOT}")
    endif()
else()
    # Locate byproducts
    set(lib_dir lib)
    # Setup download links
    if(WIN32)
        set(FILAMENT_URL https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.15.1-windows.tgz)
        set(FILAMENT_SHA256 cf073720231b01c13c173a71681c170c5eaf1ef89d52cb4ad49d94f0192f9e2f)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.15.1/filament-v1.15.1-mac.tgz)
        set(FILAMENT_SHA256 2c48431c37d1f9c67c81949441f3d6f593c495c235525b32e4e67f826c268ca9)
        string(APPEND lib_dir /x86_64)
    else() # Linux
        set(FILAMENT_URL https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.15.1-linux.tgz)
        set(FILAMENT_SHA256 5e77765885ff22a9528cdd8ac262d53c7b2e9f7a3381f7b39ee7bf8b6ef76e9b)
        string(APPEND lib_dir /x86_64)
    endif()

    set(lib_byproducts ${filament_LIBRARIES})
    list(TRANSFORM lib_byproducts PREPEND <SOURCE_DIR>/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
    list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})

    if(WIN32)
        set(lib_byproducts_debug ${filament_LIBRARIES})
        list(TRANSFORM lib_byproducts_debug PREPEND <SOURCE_DIR>/${lib_dir}d/${CMAKE_STATIC_LIBRARY_PREFIX})
        list(TRANSFORM lib_byproducts_debug APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
        list(APPEND lib_byproducts ${lib_byproducts_debug})
    endif()

    # ExternalProject_Add happens at build time.
    ExternalProject_Add(
        ext_filament
        PREFIX filament
        URL ${FILAMENT_URL}
        URL_HASH SHA256=${FILAMENT_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/filament"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE ON
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${lib_byproducts}
    )
    ExternalProject_Get_Property(ext_filament SOURCE_DIR)
    set(FILAMENT_ROOT ${SOURCE_DIR})
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")
