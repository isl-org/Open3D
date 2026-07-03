# CMake file for consuming pre-compiled WebRTC.
#
# See 3rdparty/webrtc/README.md (Method 1) for more information.

include(ExternalProject)

set(WEBRTC_VER e8b4d4c)
set(WEBRTC_USE_LOCAL_ARCHIVE OFF)
if(DEFINED ENV{OPEN3D_WEBRTC_PREBUILT_ARCHIVE} AND NOT "$ENV{OPEN3D_WEBRTC_PREBUILT_ARCHIVE}" STREQUAL "")
    set(WEBRTC_URL "file://$ENV{OPEN3D_WEBRTC_PREBUILT_ARCHIVE}")
    set(WEBRTC_USE_LOCAL_ARCHIVE ON)
endif()
if (APPLE)
    if(NOT WEBRTC_URL)
        set(WEBRTC_URL
            https://github.com/isl-org/open3d_downloads/releases/download/webrtc-v4/webrtc_${WEBRTC_VER}_macos_arm64.tar.gz
        )
    endif()
    # Update after publishing M149 macOS arm64 artifact from webrtc.yml.
    set(WEBRTC_SHA256 PLACEHOLDER_MACOS_ARM64_SHA256)
elseif (WIN32)
    # Prebuilt WebRTC is a static lib for both MSVC runtimes. A shared Open3D
    # build links it into open3d.dll and always uses the dynamic runtime, so
    # only BUILD_SHARED_LIBS=ON + STATIC_WINDOWS_RUNTIME=ON is unsupported.
    if (BUILD_SHARED_LIBS AND STATIC_WINDOWS_RUNTIME)
        message(FATAL_ERROR "Pre-built WebRTC does not support "
            "BUILD_SHARED_LIBS=ON with STATIC_WINDOWS_RUNTIME=ON. Use "
            "STATIC_WINDOWS_RUNTIME=OFF or BUILD_WEBRTC_FROM_SOURCE=ON.")
    endif()
    if(STATIC_WINDOWS_RUNTIME)
        set(WEBRTC_RUNTIME_TAG mt)
    else()
        set(WEBRTC_RUNTIME_TAG md)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        set(WEBRTC_CONFIG_TAG Debug)
    else()
        set(WEBRTC_CONFIG_TAG Release)
    endif()
    if(NOT WEBRTC_URL)
        set(WEBRTC_URL
            https://github.com/isl-org/open3d_downloads/releases/download/webrtc-v4/webrtc_${WEBRTC_VER}_win_${WEBRTC_CONFIG_TAG}_${WEBRTC_RUNTIME_TAG}.zip
        )
    endif()
    # Update after publishing four Windows artifacts from webrtc.yml.
    set(WEBRTC_SHA256 PLACEHOLDER_WIN_SHA256)
else()  # Linux
    if(NOT WEBRTC_URL)
        if(NOT GLIBCXX_USE_CXX11_ABI)
            message(FATAL_ERROR "Pre-built WebRTC with GLIBCXX_USE_CXX11_ABI=OFF is "
                "no longer provided. Use GLIBCXX_USE_CXX11_ABI=ON or "
                "BUILD_WEBRTC_FROM_SOURCE=ON.")
        endif()
        set(WEBRTC_URL
            https://github.com/isl-org/open3d_downloads/releases/download/webrtc-v4/webrtc_${WEBRTC_VER}_linux_cxx-abi-1.tar.gz
        )
    endif()
    set(WEBRTC_SHA256 1b529bf448d5abd07ec1f8d310ee5c94bd79e84fe563ae1562420f8e478cc202)
endif()

if(WEBRTC_SHA256 MATCHES "^PLACEHOLDER")
    if(WEBRTC_USE_LOCAL_ARCHIVE)
        message(WARNING "WebRTC prebuilt SHA256 not set for this platform (${WEBRTC_SHA256}). "
            "Using local OPEN3D_WEBRTC_PREBUILT_ARCHIVE without URL_HASH verification.")
        unset(WEBRTC_SHA256)
    else()
        message(FATAL_ERROR "WebRTC prebuilt SHA256 not set for this platform (${WEBRTC_SHA256}). "
            "Update webrtc_download.cmake after publishing the archive, or set "
            "OPEN3D_WEBRTC_PREBUILT_ARCHIVE to a verified local file.")
    endif()
endif()

set(_webrtc_url_hash "")
if(WEBRTC_SHA256)
    set(_webrtc_url_hash URL_HASH SHA256=${WEBRTC_SHA256})
endif()

ExternalProject_Add(
    ext_webrtc
    PREFIX webrtc
    URL ${WEBRTC_URL}
    ${_webrtc_url_hash}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/webrtc"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ""
)

ExternalProject_Get_Property(ext_webrtc SOURCE_DIR)
# Prebuilt layout: flat include/ and lib/ at archive root (M149 packages).
set(WEBRTC_PREBUILT_ROOT ${SOURCE_DIR})

# Variables consumed by find_dependencies.cmake
set(WEBRTC_INCLUDE_DIRS
    ${WEBRTC_PREBUILT_ROOT}/include/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/abseil-cpp/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/jsoncpp/source/include/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/jsoncpp/generated/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/libyuv/include/
)
set(WEBRTC_LIB_DIR ${WEBRTC_PREBUILT_ROOT}/lib)
set(WEBRTC_LIBRARIES
    webrtc
    webrtc_extra
)

# Dummy target that depends on all WebRTC targets.
add_custom_target(ext_webrtc_all)
add_dependencies(ext_webrtc_all ext_webrtc)
