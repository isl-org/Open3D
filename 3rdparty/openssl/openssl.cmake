include(ExternalProject)

set(BORINGSSL_VER edfe413)
if (APPLE)
    set(BORINGSSL_URL
        https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_${BORINGSSL_VER}_darwin_x86_64.tar.gz
    )
    set(BORINGSSL_SHA256 6d5d6372d9c448d2443eb52215993912ddc122aa6fa4f8ef6d643a4480e1854b)
elseif (WIN32)
    set(BORINGSSL_URL
        https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_${BORINGSSL_VER}_win_amd64.tar.gz
    )
    set(BORINGSSL_SHA256 3baeffb18f2cb3ec9674cbe85415abcdc7a23bc9289aafc6e28eb8656f67136c)
else()  # Linux
    set(BORINGSSL_URL
        https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/boringssl_${BORINGSSL_VER}_linux_x86_64.tar.gz
    )
    set(BORINGSSL_SHA256 9ad9bab54d8872da9957535c15571d4976c14816f86675de3b15cf35ddb92056)
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
