include(ExternalProject)

include(ProcessorCount)
ProcessorCount(NPROC)

ExternalProject_Add(
    ext_openssl
    PREFIX openssl
    URL https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1k.tar.gz
    URL_HASH SHA256=b92f9d3d12043c02860e5e602e50a73ed21a69947bcc74d391f41148e9f6aa95
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/openssl"
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND CC="${CMAKE_C_COMPILER}" && ./config --prefix=<INSTALL_DIR> --openssldir=<INSTALL_DIR>
    BUILD_COMMAND     CC="${CMAKE_C_COMPILER}" && make -j${NPROC}
    INSTALL_COMMAND   CC="${CMAKE_C_COMPILER}" && make install_sw
)

set(OPENSSL_BUILD_BYPRODUCTS "<INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ssl${CMAKE_STATIC_LIBRARY_SUFFIX}"
                             "<INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}crypto${CMAKE_STATIC_LIBRARY_SUFFIX}"
)

ExternalProject_Get_Property(ext_openssl INSTALL_DIR)
set(OPENSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/ ${INSTALL_DIR}/src/ext_openssl/) # "/" is critical.
set(OPENSSL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(OPENSSL_LIBRARIES ssl crypto)
