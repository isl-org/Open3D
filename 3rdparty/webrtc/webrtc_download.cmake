# CMake file for consuming pre-compiled WebRTC.
#
# See 3rdparty/webrtc/README.md (Method 1) for more information.

include(ExternalProject)

set(WEBRTC_VER e8b4d4c)
set(WEBRTC_BASE_URL
    https://github.com/isl-org/open3d_downloads/releases/download/webrtc-v4)

# Windows prebuilt WebRTC ships four variants (Debug / Release x /MT[d] //
# MD[d] runtime). Visual Studio (a multi-config generator) selects Debug vs.
# Release at *build* time (`cmake --build . --config ...`), not at CMake
# configure time, so CMAKE_BUILD_TYPE cannot be used to decide which single
# variant to download. For multi-config generators, the STATIC_WINDOWS_RUNTIME
# choice (mt vs md) is still fixed at configure time, but the Debug/Release
# pick is deferred to build time via a `$<CONFIG>` generator expression, so
# only the variant actually built is ever downloaded (see the
# add_custom_command below). Single-config generators (Ninja, Makefiles)
# resolve to one variant now, defaulting to Release if CMAKE_BUILD_TYPE is
# unset.
get_property(WEBRTC_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if (APPLE)
    set(WEBRTC_URL
        ${WEBRTC_BASE_URL}/webrtc_${WEBRTC_VER}_macos_arm64.tar.gz
    )
    set(WEBRTC_SHA256 3c2592a3bd9efcee591924007857342fec3753e2ae68695baeb2b8774f0e3abc)
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
        set(WEBRTC_DEBUG_TAG Debug_mt)
        set(WEBRTC_DEBUG_SHA256 b537cce72f758fbcd214fbca80ecfb26228d23c728119abc499c4b333e5f0786)
        set(WEBRTC_RELEASE_TAG Release_mt)
        set(WEBRTC_RELEASE_SHA256 99e7aabfa38a9ce276f606e461feb7c655771095f8e6b14508985dfa44a4bc3a)
    else()
        set(WEBRTC_DEBUG_TAG Debug_md)
        set(WEBRTC_DEBUG_SHA256 b59fb3cd16eaf022865f39fcd4aa25333f0ff74923b62c62acee4c801431e6fe)
        set(WEBRTC_RELEASE_TAG Release_md)
        set(WEBRTC_RELEASE_SHA256 25339a82017e2614f3c2a9bfcbb0895a51532aaadeba0035c6ac266c154f03c9)
    endif()
    set(WEBRTC_DEBUG_URL
        ${WEBRTC_BASE_URL}/webrtc_${WEBRTC_VER}_win_${WEBRTC_DEBUG_TAG}.zip)
    set(WEBRTC_RELEASE_URL
        ${WEBRTC_BASE_URL}/webrtc_${WEBRTC_VER}_win_${WEBRTC_RELEASE_TAG}.zip)
    if(NOT WEBRTC_MULTI_CONFIG)
        # Single-config generator: resolve to one variant now, as with the
        # other platforms below.
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE Release)
        endif()
        if(CMAKE_BUILD_TYPE STREQUAL Debug)
            set(WEBRTC_URL ${WEBRTC_DEBUG_URL})
            set(WEBRTC_SHA256 ${WEBRTC_DEBUG_SHA256})
        else()
            set(WEBRTC_URL ${WEBRTC_RELEASE_URL})
            set(WEBRTC_SHA256 ${WEBRTC_RELEASE_SHA256})
        endif()
    endif()
else()  # Linux
    if(NOT GLIBCXX_USE_CXX11_ABI)
        message(FATAL_ERROR "Pre-built WebRTC with GLIBCXX_USE_CXX11_ABI=OFF is "
            "no longer provided. Use GLIBCXX_USE_CXX11_ABI=ON or "
            "BUILD_WEBRTC_FROM_SOURCE=ON.")
    endif()
    set(WEBRTC_URL
        ${WEBRTC_BASE_URL}/webrtc_${WEBRTC_VER}_linux_cxx-abi-1.tar.xz
    )
    set(WEBRTC_SHA256 0209f722974fa7b9da9d1c8e279694cf2c4db81e079a325bcadb1b6e0c3b6981) # 1b529bf448d5abd07ec1f8d310ee5c94bd79e84fe563ae1562420f8e478cc202
endif()

if(WIN32 AND WEBRTC_MULTI_CONFIG)
    # Multi-config (Visual Studio): don't download both variants up front.
    # ExternalProject_Add's URL/URL_HASH are resolved once at configure time
    # into a fixed download script, so they can't pick a variant based on
    # $<CONFIG>. Instead, drive the download from a plain add_custom_command
    # whose OUTPUT path embeds $<CONFIG>; multi-config generators only build
    # the outputs needed for the config actually requested
    # (`cmake --build . --config ...`), so only one variant -- the one
    # actually built -- ever gets downloaded.
    set(WEBRTC_PREBUILT_ROOT "${CMAKE_BINARY_DIR}/webrtc/$<CONFIG>")
    set(WEBRTC_STAMP "${WEBRTC_PREBUILT_ROOT}/webrtc_fetch.stamp")
    add_custom_command(
        OUTPUT "${WEBRTC_STAMP}"
        COMMAND ${CMAKE_COMMAND}
            "-DURL=$<IF:$<CONFIG:Debug>,${WEBRTC_DEBUG_URL},${WEBRTC_RELEASE_URL}>"
            "-DSHA256=$<IF:$<CONFIG:Debug>,${WEBRTC_DEBUG_SHA256},${WEBRTC_RELEASE_SHA256}>"
            "-DDEST=${WEBRTC_PREBUILT_ROOT}"
            "-DSTAMP=${WEBRTC_STAMP}"
            -P "${CMAKE_CURRENT_LIST_DIR}/webrtc_fetch_variant.cmake"
        COMMENT "Downloading prebuilt WebRTC ($<CONFIG>)"
        VERBATIM
    )
    add_custom_target(ext_webrtc_all DEPENDS "${WEBRTC_STAMP}")
    set(WEBRTC_LIB_DIR "${WEBRTC_PREBUILT_ROOT}/lib")
else()
    ExternalProject_Add(
        ext_webrtc
        PREFIX webrtc
        URL ${WEBRTC_URL}
        URL_HASH SHA256=${WEBRTC_SHA256}
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
    set(WEBRTC_LIB_DIR ${WEBRTC_PREBUILT_ROOT}/lib)

    add_custom_target(ext_webrtc_all)
    add_dependencies(ext_webrtc_all ext_webrtc)
endif()

# Variables consumed by find_dependencies.cmake
set(WEBRTC_INCLUDE_DIRS
    ${WEBRTC_PREBUILT_ROOT}/include/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/abseil-cpp/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/jsoncpp/source/include/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/jsoncpp/generated/
    ${WEBRTC_PREBUILT_ROOT}/include/third_party/libyuv/include/
)
set(WEBRTC_LIBRARIES
    webrtc
    webrtc_extra
)
