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
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.58.2/filament-windows.tgz)
        set(FILAMENT_SHA256 8b07ea0f234a3eececed8e9fda9742de48ab04260aab2aa472d70242d790c8db)
        if (STATIC_WINDOWS_RUNTIME)
            string(APPEND lib_dir /x86_64/mt)
        else()
            string(APPEND lib_dir /x86_64/md)
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.58.2/filament-v1.58.2-mac.tgz)
        set(FILAMENT_SHA256 d0fceb641295756fa674520c9ebf19f8929278b5fc550e5abbf50c3094e6922a)
    else()      
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.58.2/filament-v1.58.2-linux.tgz)
        set(FILAMENT_SHA256 9c4c715402f0cb675660c34f2fdf3d00f12b198b42bfbf0765a3ec40f33c84cc)
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