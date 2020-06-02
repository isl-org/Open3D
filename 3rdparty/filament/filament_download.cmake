if (FILAMENT_PRECOMPILED_ROOT)
    if (EXISTS "${FILAMENT_PRECOMPILED_ROOT}")
        set(FILAMENT_ROOT "${FILAMENT_PRECOMPILED_ROOT}")
    else()
        message(FATAL_ERROR "Filament binaries not found in ${FILAMENT_PRECOMPILED_ROOT}")
    endif()
else()
    set(FILAMENT_ROOT ${CMAKE_BINARY_DIR}/downloads/filament)

    if (USE_VULKAN AND (ANDROID OR WIN32 OR WEBGL OR IOS))
        MESSAGE(FATAL_ERROR "Downloadable version of Filament supports vulkan only on Linux and Apple")
    endif()

    if (NOT EXISTS ${FILAMENT_ROOT}/README.md)
        set(DOWNLOAD_PATH ${CMAKE_BINARY_DIR}/downloads)
        set(TAR_PWD ${DOWNLOAD_PATH})

        if (NOT EXISTS ${ARCHIVE_FILE})
            set(ARCHIVE_FILE ${CMAKE_BINARY_DIR}/downloads/filament.tgz)

            # Setup download links ============================================================================
            set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200220-linux.tgz")
            set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-linux.tgz")

            if (WIN32)
                set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200127-windows.tgz")
                set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-windows.tgz")
                
                file(MAKE_DIRECTORY ${FILAMENT_ROOT})
                set(TAR_PWD ${FILAMENT_ROOT})
            elseif (APPLE)
                set(DOWNLOAD_URL_PRIMARY "https://storage.googleapis.com/isl-datasets/open3d-dev/filament-20200127-mac-10.14-resizefix2.tgz")
                set(DOWNLOAD_URL_FALLBACK "https://github.com/google/filament/releases/download/v1.4.5/filament-20200127-mac.tgz")
            endif()
            # =================================================================================================

            file(DOWNLOAD ${DOWNLOAD_URL_PRIMARY} ${ARCHIVE_FILE} SHOW_PROGRESS STATUS DOWNLOAD_RESULT)
            if (NOT DOWNLOAD_RESULT EQUAL 0)
                file(DOWNLOAD ${DOWNLOAD_URL_FALLBACK} ${ARCHIVE_FILE} SHOW_PROGRESS STATUS DOWNLOAD_RESULT)
            endif()
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${ARCHIVE_FILE} WORKING_DIRECTORY ${TAR_PWD})
    endif()
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")

set(filament_LIBRARIES filameshio filament filamat_lite filaflat filabridge geometry backend bluegl ibl image meshoptimizer smol-v utils)
if (UNIX)
    set(filament_LIBRARIES ${filament_LIBRARIES} bluevk)
endif()
