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
        set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200127-windows.tgz")
        set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-windows.tgz")
    elseif(APPLE)
        set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200127-mac-10.14-resizefix2.tgz")
        set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-mac.tgz")
    else()
        set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200220-linux.tgz")
        set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-linux.tgz")
    endif()

    if (USE_VULKAN AND (ANDROID OR WIN32 OR WEBGL OR IOS))
        MESSAGE(FATAL_ERROR "Downloadable version of Filament supports vulkan only on Linux and Apple")
    endif()

    FetchContent_Declare(
        fetch_filament
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
    )

    # FetchContent happends at config time.
    FetchContent_GetProperties(fetch_filament)
    if(NOT fetch_filament_POPULATED)
        message(STATUS "Downloading Filament...")
        FetchContent_Populate(fetch_filament)
        # We use the default download and unpack directories for FetchContent.
        message(STATUS "Filament has been downloaded to ${fetch_filament_DOWNLOADED_FILE}.")
        message(STATUS "Filament has been extracted to ${fetch_filament_SOURCE_DIR}.")
        set(FILAMENT_ROOT "${fetch_filament_SOURCE_DIR}")
    endif()
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")

set(filament_LIBRARIES filameshio filament filamat_lite filaflat filabridge geometry backend bluegl ibl image meshoptimizer smol-v utils)
if (UNIX)
    set(filament_LIBRARIES ${filament_LIBRARIES} bluevk)
endif()
