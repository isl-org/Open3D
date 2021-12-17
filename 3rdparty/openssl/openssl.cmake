include(ExternalProject)

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
    set(BORINGSSL_SHA256 3baeffb18f2cb3ec9674cbe85415abcdc7a23bc9289aafc6e28eb8656f67136c)
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
    ext_openssl
    PREFIX openssl
    URL ${BORINGSSL_URL}
    URL_HASH SHA256=${BORINGSSL_SHA256}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/openssl"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ""
)

ExternalProject_Get_Property(ext_openssl SOURCE_DIR)
set(OPENSSL_ROOT_DIR_FOR_CURL ${SOURCE_DIR})
set(OPENSSL_INCLUDE_DIRS ${OPENSSL_ROOT_DIR_FOR_CURL}/include/) # "/" is critical.

if(BUILD_WEBRTC)
    set(OPENSSL_LIB_DIR)   # Empty, as we use the sysmbols from WebRTC.
    set(OPENSSL_LIBRARIES) # Empty, as we use the sysmbols from WebRTC.
else()
    set(OPENSSL_LIB_DIR ${OPENSSL_ROOT_DIR_FOR_CURL}/lib)
    set(OPENSSL_LIBRARIES ssl crypto)
endif()
