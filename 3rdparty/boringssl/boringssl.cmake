include(ExternalProject)

# Since boringssl requires additional dependencies, we provide prebuilt
# boringssl for all platforms. The compilation is done by running:
#
# - build_boringssl.ps1: fow Windows
# - build_boringssl.sh : for Ubuntu/macOS
#
# boringssl_commit in build_boringssl.xx should be consistent with the boringssl
# used by WEBRTC_COMMIT in 3rdparty/webrtc/webrtc_build.sh
#
# 1. Clone the webrtc repo, as described in 3rdparty/webrtc/README.md
# 2. Checkout the `WEBRTC_COMMIT`
# 3. Run `cd src/third_party` and `git log --name-status -10 boringssl`
# 4. Read and understand the commit log. The commit in the boringssl repo is not
#    the same as the one used in `third_party`, as the `third_party` repo copies
#    the boringssl code and does not use a submodule. Determine the commit id
#    of boringssl manually.

if(APPLE)
    if(APPLE_AARCH64)
        set(BORINGSSL_URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_edfe413_darwin_arm64.tar.gz)
        set(BORINGSSL_SHA256 d0a574782cd1fa1ef4c02a57b96ea7aafef30a3a77b2163c02e9854ad8964d5e)
    else()
        set(BORINGSSL_URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_edfe413_darwin_x86_64.tar.gz)
        set(BORINGSSL_SHA256 6d5d6372d9c448d2443eb52215993912ddc122aa6fa4f8ef6d643a4480e1854b)
    endif()
elseif(WIN32)
    set(BORINGSSL_URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_edfe413_win_amd64.tar.gz)
    set(BORINGSSL_SHA256 fd538d545990a4657ee2b22c444e0baf61edaa1609f84cdfc9217659c44988c4)
else()
    if(LINUX_AARCH64)
        set(BORINGSSL_URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_edfe413_linux_aarch64.tar.gz)
        set(BORINGSSL_SHA256 5f10d8d529094aa8b6d40f7caaead9652a8b4ce7dd2b48e186d1b519a66efd2c)
    else()
        set(BORINGSSL_URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_edfe413_linux_x86_64.tar.gz)
        set(BORINGSSL_SHA256 9ad9bab54d8872da9957535c15571d4976c14816f86675de3b15cf35ddb92056)
    endif()
endif()

ExternalProject_Add(
    ext_boringssl
    PREFIX boringssl
    URL ${BORINGSSL_URL}
    URL_HASH SHA256=${BORINGSSL_SHA256}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/boringssl"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ""
)

ExternalProject_Get_Property(ext_boringssl SOURCE_DIR)
if(WIN32)
    set(BORINGSSL_ROOT_DIR ${SOURCE_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>)
else()
    set(BORINGSSL_ROOT_DIR ${SOURCE_DIR})
endif()
set(BORINGSSL_INCLUDE_DIRS ${BORINGSSL_ROOT_DIR}/include/) # "/" is critical.

if(BUILD_WEBRTC)
    set(BORINGSSL_LIB_DIR)   # Empty, as we use the sysmbols from WebRTC.
    set(BORINGSSL_LIBRARIES) # Empty, as we use the sysmbols from WebRTC.
else()
    set(BORINGSSL_LIB_DIR ${BORINGSSL_ROOT_DIR}/lib)
    set(BORINGSSL_LIBRARIES ssl crypto)
endif()
