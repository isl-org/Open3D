include(ExternalProject)

set(filament_LIBRARIES filameshio filament filaflat filabridge geometry backend bluegl bluevk ibl image ktxreader meshoptimizer smol-v utils shaders)
if(NOT DEFINED FILAMENT_VULKAN_EXTERNAL_IMAGE_IMPORT)
    set(FILAMENT_VULKAN_EXTERNAL_IMAGE_IMPORT OFF)
endif()

if (FILAMENT_PRECOMPILED_ROOT)
    if (EXISTS "${FILAMENT_PRECOMPILED_ROOT}")
        set(FILAMENT_ROOT "${FILAMENT_PRECOMPILED_ROOT}")
    else()
        message(FATAL_ERROR "Filament binaries not found in ${FILAMENT_PRECOMPILED_ROOT}")
    endif()
else()
    # Locate byproducts
    set(lib_dir lib)
    get_property(FILAMENT_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    # Setup download links
    if(WIN32)
        set(FILAMENT_BASE_URL https://github.com/isl-org/open3d_downloads/releases/download/filament-v1.76)
        set(FILAMENT_VULKAN_EXTERNAL_IMAGE_IMPORT ON)
        if (STATIC_WINDOWS_RUNTIME)
            set(FILAMENT_RELEASE_TAG mt)
            set(FILAMENT_RELEASE_SHA256 955CD3634C670A30FC9FB46FF463BFFEC2D65765442EB9A7C1AFA8006C265BBD)
            set(FILAMENT_DEBUG_TAG mtd)
            set(FILAMENT_DEBUG_SHA256 C332F1282ACC7DE6BDECDECAFF4305F7F963775DD3BA73389C178CDBCC8526FD)
        else()
            set(FILAMENT_RELEASE_TAG md)
            set(FILAMENT_RELEASE_SHA256 C7514C39237AD441C93119B4FB211F584C56EED124B2BECCBB399B1B7823E3C1)
            set(FILAMENT_DEBUG_TAG mdd)
            set(FILAMENT_DEBUG_SHA256 C018D06E50BDA8587934D39FDEDF3308812803D40B9D63ED00C262D71338B9D8)
        endif()
        set(FILAMENT_RELEASE_URL
            ${FILAMENT_BASE_URL}/filament-v1.76.0-windows-msvc-x64-Release_${FILAMENT_RELEASE_TAG}.zip)
        set(FILAMENT_DEBUG_URL
            ${FILAMENT_BASE_URL}/filament-v1.76.0-windows-msvc-x64-RelWithDebInfo_${FILAMENT_RELEASE_TAG}.zip)
        if(NOT FILAMENT_MULTI_CONFIG)
            if(CMAKE_BUILD_TYPE STREQUAL Debug OR CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
                set(FILAMENT_URL ${FILAMENT_DEBUG_URL})
                set(FILAMENT_SHA256 ${FILAMENT_DEBUG_SHA256})
                string(APPEND lib_dir "/x86_64/${FILAMENT_DEBUG_TAG}")
            else()
                set(FILAMENT_URL ${FILAMENT_RELEASE_URL})
                set(FILAMENT_SHA256 ${FILAMENT_RELEASE_SHA256})
                string(APPEND lib_dir "/x86_64/${FILAMENT_RELEASE_TAG}")
            endif()
        endif()
    elseif(APPLE)
        set(FILAMENT_URL https://github.com/google/filament/releases/download/v1.76.0/filament-v1.76.0-mac.tgz)
        set(FILAMENT_SHA256 6f067ac0931b305c32be108679cf5b0c59a3fb51753d0c64d60d290b4f28b2db)
    else()
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
            set(FILAMENT_URL
                https://github.com/isl-org/open3d_downloads/releases/download/filament-v1.76/filament-v1.76.0-linux-22.04.tgz)
            set(FILAMENT_SHA256 ad0c349bba319012785b85c1ef978c633600ef1da3c0ae97b064a828a243adbf)
            set(FILAMENT_VULKAN_EXTERNAL_IMAGE_IMPORT ON)
            message(STATUS "Using Open3D patched Filament binary for Linux x86_64.")
        else()
            set(FILAMENT_URL
                    https://github.com/google/filament/releases/download/v1.76.0/filament-v1.76.0-linux.tgz)
            set(FILAMENT_SHA256 08f96fbce1432d7a5faf34b3e96a186639b89663f8a215e6d2c36ad6cb73fa4a)
            message(STATUS "Using upstream Filament binary for Linux ${CMAKE_SYSTEM_PROCESSOR}. Gaussian Splat rendering will not be available.")
        endif()
    endif()

    if(WIN32 AND FILAMENT_MULTI_CONFIG)
        set(FILAMENT_ROOT "${CMAKE_BINARY_DIR}/filament/$<CONFIG>")
        set(FILAMENT_STAMP "${FILAMENT_ROOT}/filament_fetch.stamp")
        set(FILAMENT_LIB_DIR "${FILAMENT_ROOT}/lib/x86_64/$<IF:$<CONFIG:Debug>,${FILAMENT_DEBUG_TAG},${FILAMENT_RELEASE_TAG}>")
        set(FILAMENT_BYPRODUCTS ${filament_LIBRARIES})
        list(TRANSFORM FILAMENT_BYPRODUCTS PREPEND "${FILAMENT_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}")
        list(TRANSFORM FILAMENT_BYPRODUCTS APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
        add_custom_command(
            OUTPUT "${FILAMENT_STAMP}"
            COMMAND ${CMAKE_COMMAND}
                "-DURL=$<IF:$<CONFIG:Debug>,${FILAMENT_DEBUG_URL},${FILAMENT_RELEASE_URL}>"
                "-DSHA256=$<IF:$<CONFIG:Debug>,${FILAMENT_DEBUG_SHA256},${FILAMENT_RELEASE_SHA256}>"
                "-DCACHE_DIR=${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/filament"
                "-DDEST=${FILAMENT_ROOT}"
                "-DSTAMP=${FILAMENT_STAMP}"
                -P "${CMAKE_CURRENT_LIST_DIR}/filament_fetch_variant.cmake"
            BYPRODUCTS ${FILAMENT_BYPRODUCTS}
            COMMENT "Downloading prebuilt Filament ($<CONFIG>)"
            VERBATIM
        )
        add_custom_target(ext_filament DEPENDS "${FILAMENT_STAMP}")
    else()
        set(lib_byproducts ${filament_LIBRARIES})
        list(TRANSFORM lib_byproducts PREPEND <SOURCE_DIR>/${lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX})
        list(TRANSFORM lib_byproducts APPEND ${CMAKE_STATIC_LIBRARY_SUFFIX})
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
endif()

message(STATUS "Filament is located at ${FILAMENT_ROOT}")