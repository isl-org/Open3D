include(ExternalProject)

ExternalProject_Add(
    ext_boringssl
    PREFIX boringssl
    URL https://github.com/google/boringssl/archive/refs/heads/master.zip
    URL_HASH SHA256=54d51f4873d28d5e1eb6f99a37625d3d47b1ccd1f0a93453f3871f8b5c8ad207
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/boringssl"
    CMAKE_ARGS ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target ssl crypto
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include
    COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/ssl/libssl.a <INSTALL_DIR>/lib/libssl.a
    COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/crypto/libcrypto.a <INSTALL_DIR>/lib/libcrypto.a
)

set(BORINGSSL_BUILD_BYPRODUCTS "<INSTALL_DIR>/src/ext_boringssl-build/ssl/libssl.a"
                               "<INSTALL_DIR>/src/ext_boringssl-build/crypto/libcryp.a"
)

ExternalProject_Get_Property(ext_boringssl INSTALL_DIR)
message(STATUS "BORING_SSL_INSTALL_DIR ${INSTALL_DIR}")

ExternalProject_Get_Property(ext_boringssl SOURCE_DIR)
message(STATUS "BORING_SSL_SOURCE_DIR ${SOURCE_DIR}")

set(BORINGSSL_INSTALL_DIR ${INSTALL_DIR})
set(BORINGSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(BORINGSSL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(BORINGSSL_LIBRARIES ssl crypto)
