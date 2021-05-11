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
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.9.9/filament-v1.9.9-windows.tgz")
        # Required for filament v1.9.9
        # Older versions of filament do not contain vkshaders.
        # They also have a different directory structure.
        # Remove vkshaders here so we can use newer versions with FILAMENT_PRECOMPILED_ROOT.
        list(REMOVE_ITEM filament_LIBRARIES vkshaders)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.9.19/filament-v1.9.19-mac.tgz")
        string(APPEND lib_dir /x86_64)
    else()      # Linux: Check glibc version and use open3d filament binary if new (Ubuntu 20.04 and similar)
        execute_process(COMMAND ldd --version OUTPUT_VARIABLE ldd_version)
        string(REGEX MATCH "([0-9]+\.)+[0-9]+" glibc_version ${ldd_version})
        if(${glibc_version} VERSION_LESS "2.31")
            set(DOWNLOAD_URL_PRIMARY
                "https://github.com/intel-isl/open3d_downloads/releases/download/filament/filament-v1.9.19-linux.tgz")
            message(STATUS "GLIBC version ${glibc_version} found: Downloading "
                "Google Filament binary.")
        else()
            set(DOWNLOAD_URL_PRIMARY
                "https://github.com/intel-isl/open3d_downloads/releases/download/filament/filament-v1.9.19-linux-20.04.tgz")
            message(STATUS "GLIBC version ${glibc_version} found: Downloading "
                "Open3D Filament binary.")
        endif()
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
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
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
