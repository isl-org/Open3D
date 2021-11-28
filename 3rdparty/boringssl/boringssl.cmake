include(ExternalProject)

set(BORINGSSL_VER b3ed07)
if (APPLE)

elseif (WIN32)

else()  # Linux
    set(BORINGSSL_URL
        https://github.com/isl-org/open3d_downloads/releases/download/boringssl-v1/boringssl_${BORINGSSL_VER}_linux_x64.tar.xz
    )
    set(BORINGSSL_SHA256 7197b47968f4fb1a68b5f2003b3464dd715bb3a48a5ba68e483233206f6cc94d)
endif()


if(MSVC)
    set(lib_name boringssl_static)
else()
    set(lib_name boringssl)
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
message(STATUS "BoringSSL source dir: ${SOURCE_DIR}")

set(BORINGSSL_INCLUDE_DIRS
    ${SOURCE_DIR}/include/
)
set(BORINGSSL_LIB_DIR ${SOURCE_DIR}/build/lib/)
set(BORINGSSL_LIBRARIES
    ssl
    crypto
)
