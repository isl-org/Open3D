include(FetchContent)

set(filament_LIBRARIES filameshio filament filaflat filabridge geometry backend bluegl bluevk ibl image ktxreader meshoptimizer smol-v utils vkshaders)

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
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.58.0/filament-v1.58.0-windows.tgz)
        set(FILAMENT_SHA256 b1a018cd6380abd345661dbb22bf13b8ebb86c6df8672267615ca2aa724aa1d0)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.58.0/filament-v1.58.0-mac.tgz)
        set(FILAMENT_SHA256 810b37b35716432ffefa7a262b005201dab4773393c15caa946914df88058cab)
    else()      # Linux: Check glibc version and use open3d filament binary if new (Ubuntu 20.04 and similar)
        execute_process(COMMAND ldd --version OUTPUT_VARIABLE ldd_version)
        string(REGEX MATCH "([0-9]+\.)+[0-9]+" glibc_version ${ldd_version})
        if(${glibc_version} VERSION_LESS "2.33")
            set(FILAMENT_URL
                    https://github.com/isl-org/open3d_downloads/releases/download/filament/filament-v1.49.1-ubuntu20.04.tgz)
            set(FILAMENT_SHA256 f4ba020f0ca63540e2f86b36d1728a1ea063ddd5eb55b0ba6fc621ee815a60a7)
            message(STATUS "GLIBC version ${glibc_version} found: Using "
                    "Open3D built Filament binary for Ubuntu 20.04.")
        else()
            set(FILAMENT_URL
                    https://github.com/google/filament/releases/download/v1.58.0/filament-v1.58.0-linux.tgz)
            set(FILAMENT_SHA256 71a0ffd9577ae510151792c28d9ba841d67e20757730dae00ecd5b0b3e108b79)
            message(STATUS "GLIBC version ${glibc_version} found: Using "
                    "Google Filament binary.")
        endif()
    endif()

    set(lib_byproducts ${filament_LIBRARIES})
    list(TRANSFORM lib_byproducts PREPEND <SOURCE_DIR>/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
    list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
    message(STATUS "Filament byproducts: ${lib_byproducts}")

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
    message(STATUS "Filament source dir is ${SOURCE_DIR}")
    set(FILAMENT_ROOT ${SOURCE_DIR})
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")