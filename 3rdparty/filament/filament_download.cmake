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
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.9.5/filament-v1.9.5-windows.tgz")
        set(FILAMENT_SHA256 44bb7449df41bae5f92462da086df3b2135b71503cac142b7ecd33cf4549fd86)
    elseif(APPLE)
        set(DOWNLOAD_URL_PRIMARY "https://github.com/google/filament/releases/download/v1.9.5/filament-v1.9.5-mac.tgz")
        set(FILAMENT_SHA256 c2dd644546cd8b03497e90a4e705501eb5b0b485054a6bc6063cd786fef25252)
    else()      # Linux: Check glibc version and use open3d filament binary if new (Ubuntu 20.04 and similar)
        execute_process(COMMAND ldd --version OUTPUT_VARIABLE ldd_version)
        string(REGEX MATCH "([0-9]+\.)+[0-9]+" glibc_version ${ldd_version})
        if(${glibc_version} VERSION_LESS "2.31")
            set(DOWNLOAD_URL_PRIMARY
                "https://github.com/google/filament/releases/download/v1.9.5/filament-v1.9.5-linux.tgz")
            set(FILAMENT_SHA256 d63841440b36fc5d76a99716433860262b2400eaad22835e2afc2d254932dcd5)
            message(STATUS "GLIBC version ${glibc_version} found: Downloading "
                "Google Filament binary.")
        else()
            set(DOWNLOAD_URL_PRIMARY
                "https://storage.googleapis.com/open3d-releases/filament/filament-v1.9.5-linux.tgz")
            set(FILAMENT_SHA256 2bcc4ea286bb512628caf7a67bd6687e6b50dd0c96f1d685fb8fc7498419eb44)
            message(STATUS "GLIBC version ${glibc_version} found: Downloading "
                "Open3D Filament binary.")
        endif()
    endif()

    # ExternalProject_Add happens at build time.
    ExternalProject_Add(
        ext_filament
        PREFIX filament
        URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
        URL_HASH SHA256=${FILAMENT_SHA256}
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
