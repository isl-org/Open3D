include(ExternalProject)

set(libssl_name    ${CMAKE_STATIC_LIBRARY_PREFIX}ssl${CMAKE_STATIC_LIBRARY_SUFFIX})
set(libcrypto_name ${CMAKE_STATIC_LIBRARY_PREFIX}crypto${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Add(
    ext_boringssl
    PREFIX boringssl
    URL https://github.com/google/boringssl/archive/refs/heads/master.zip
    URL_HASH SHA256=54d51f4873d28d5e1eb6f99a37625d3d47b1ccd1f0a93453f3871f8b5c8ad207
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/boringssl"
    CMAKE_ARGS ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target ssl crypto
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include        <INSTALL_DIR>/include
    COMMAND         ${CMAKE_COMMAND} -E copy <BINARY_DIR>/ssl/${libssl_name}       <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libssl_name}
    COMMAND         ${CMAKE_COMMAND} -E copy <BINARY_DIR>/crypto/${libcrypto_name} <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libcrypto_name}
    BUILD_BYPRODUCTS <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libssl_name}
                     <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libcrypto_name}
)

ExternalProject_Get_Property(ext_boringssl INSTALL_DIR)
message(STATUS "BORING_SSL_INSTALL_DIR ${INSTALL_DIR}")

ExternalProject_Get_Property(ext_boringssl SOURCE_DIR)
message(STATUS "BORING_SSL_SOURCE_DIR ${SOURCE_DIR}")

set(BORINGSSL_INSTALL_DIR ${INSTALL_DIR})
set(BORINGSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(BORINGSSL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(BORINGSSL_LIBRARIES ssl crypto)
