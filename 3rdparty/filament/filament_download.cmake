include(FetchContent)

if (FILAMENT_PRECOMPILED_ROOT)
    if (EXISTS "${FILAMENT_PRECOMPILED_ROOT}")
        set(FILAMENT_ROOT "${FILAMENT_PRECOMPILED_ROOT}")
    else()
        message(FATAL_ERROR "Filament binaries not found in ${FILAMENT_PRECOMPILED_ROOT}")
    endif()
else()
    # Setup download links
    if(WIN32)
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.8.1/filament-v1.8.1-windows.tgz")
    elseif(APPLE)
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.8.1/filament-v1.8.1-mac.tgz")
    else()
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.8.1/filament-v1.8.1-linux.tgz")
    endif()

    # ExternalProject_Add happends at build time.
    ExternalProject_Add(
        ext_filament
        PREFIX filament
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_IN_SOURCE ON
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    ExternalProject_Get_Property(ext_filament SOURCE_DIR)
    set(FILAMENT_ROOT ${SOURCE_DIR})
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")

set(filament_LIBRARIES filameshio filament filamat_lite filaflat filabridge geometry backend bluegl ibl image meshoptimizer smol-v utils)
if (UNIX OR WIN32)
    set(filament_LIBRARIES ${filament_LIBRARIES} bluevk)
endif()
