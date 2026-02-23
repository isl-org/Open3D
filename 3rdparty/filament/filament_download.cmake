include(FetchContent)

set(filament_LIBRARIES filameshio filament filaflat filabridge geometry backend bluegl bluevk ibl image ktxreader meshoptimizer smol-v utils shaders)

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
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.69.3/filament-v1.69.3-windows.tgz)
        set(FILAMENT_SHA256 8f6ff209a46918d2a3c627adfffbdd0c0310fc000f5aae8e64ee3bb722eaf346)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.69.3/filament-v1.69.3-mac.tgz)
        set(FILAMENT_SHA256 866f0b1a6ea5039a9f7e0b31589abfccdfbd436410fe1d779eec8068eb9724fe)
    else()      # Linux: Check glibc version to select compatible Filament binary
        execute_process(COMMAND ldd --version OUTPUT_VARIABLE ldd_version)
        string(REGEX MATCH "([0-9]+\.)+[0-9]+" glibc_version ${ldd_version})
        if(${glibc_version} VERSION_LESS "2.38")
            set(FILAMENT_URL
                    https://github.com/google/filament/releases/download/v1.54.0/filament-v1.54.0-linux.tgz)
            set(FILAMENT_SHA256 f07fbe8fcb6422a682f429d95fa2e097c538d0d900c62f0a835f595ab3909e8e)
            list(REMOVE_ITEM filament_LIBRARIES shaders)
            list(APPEND filament_LIBRARIES vkshaders)
            message(STATUS "GLIBC version ${glibc_version} found: Using "
                    "Filament v1.54.0 binary (compatible with glibc < 2.38).")
        else()
            set(FILAMENT_URL
                    https://github.com/google/filament/releases/download/v1.69.3/filament-v1.69.3-linux.tgz)
            set(FILAMENT_SHA256 2a503ff80edef11556c9729a8f3de21f72cd8fca9624ea11aeb447337f1556f7)
            message(STATUS "GLIBC version ${glibc_version} found: Using "
                    "Google Filament v1.69.3 binary.")
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