# open3d_fetch_ispc_compiler()
#
# Fetches the ISPC compiler and makes it available in the build.
# If a user-provided path is detected, fetching will be skipped.
#
# If fetched, the following variables will be defined:
# - $ENV{ISPC}
function(open3d_fetch_ispc_compiler)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "" "")

    # Check correct usage
    if (ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if (ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Missing values for arguments: ${ARG_KEYWORDS_MISSING_VALUES}")
    endif()

    # Sanity check as this module is not called from 3rdparty
    if (NOT OPEN3D_THIRD_PARTY_DOWNLOAD_DIR)
        message(FATAL_ERROR "OPEN3D_THIRD_PARTY_DOWNLOAD_DIR not defined!")
    endif()

    if (DEFINED ENV{ISPC} OR DEFINED CMAKE_ISPC_COMPILER)
        message(STATUS "Using user-provided path to the ISPC compiler")
    else()
        message(STATUS "Fetching ISPC compiler")

        include(FetchContent)

        set(ISPC_VER 1.16.1)
        if (APPLE)
            set(ISPC_URL https://github.com/ispc/ispc/releases/download/v${ISPC_VER}/ispc-v${ISPC_VER}-macOS.tar.gz)
            set(ISPC_SHA256 7dbce602d97227a9603aabfae6dc3b3aa24d1cd44f0ccfb5ae47ecd4d68e988e)
        elseif (WIN32)
            set(ISPC_URL https://github.com/ispc/ispc/releases/download/v${ISPC_VER}/ispc-v${ISPC_VER}-windows.zip)
            set(ISPC_SHA256 b34de2c36aff2afaa56b669ea41f9e614a045564ca74fc0b138e17ccea4880b7)
        else()  # Linux
            set(ISPC_URL https://github.com/ispc/ispc/releases/download/v${ISPC_VER}/ispc-v${ISPC_VER}-linux.tar.gz)
            set(ISPC_SHA256 88db3d0461147c10ed81053a561ec87d3e14265227c03318f4fcaaadc831037f)
        endif()

        FetchContent_Declare(
            ext_ispc
            PREFIX ispc
            URL ${ISPC_URL}
            URL_HASH SHA256=${ISPC_SHA256}
            DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/ispc"
        )

        FetchContent_MakeAvailable(ext_ispc)

        set(CMAKE_ISPC_COMPILER "${ext_ispc_SOURCE_DIR}/bin/ispc" PARENT_SCOPE)
    endif()
endfunction()
