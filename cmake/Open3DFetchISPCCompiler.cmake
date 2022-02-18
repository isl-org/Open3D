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

        set(ISPC_VER 1.17.0)
        if (APPLE)
            set(ISPC_URL https://github.com/ispc/ispc/releases/download/v1.17.0/ispc-v1.17.0-macOS.tar.gz)
            set(ISPC_SHA256 e7fdcdbd5c272955249148c452ccd7295d7cf77b35ca1dec377e72b49c847bff)
        elseif (WIN32)
            set(ISPC_URL https://github.com/ispc/ispc/releases/download/v1.17.0/ispc-v1.17.0-windows.zip)
            set(ISPC_SHA256 e9a7cc98f69357482985bcbf69fa006632cee7b3606069b4d5e16dc62092d660)
        else()  # Linux
            set(ISPC_URL https://github.com/ispc/ispc/releases/download/v1.17.0/ispc-v1.17.0-linux.tar.gz)
            set(ISPC_SHA256 6acc5df75efdce437f79b1b6489be8567c6d009e19dcc4851b9b37012afce1f7)
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
