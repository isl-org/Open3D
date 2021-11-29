include(ExternalProject)

if(MSVC)
    set(lib_name boringssl_static)
else()
    set(lib_name boringssl)
endif()

ExternalProject_Add(
    ext_boringssl
    PREFIX boringssl
    URL https://github.com/google/boringssl/archive/refs/heads/master.zip
    URL_HASH SHA256=54d51f4873d28d5e1eb6f99a37625d3d47b1ccd1f0a93453f3871f8b5c8ad207
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/boringssl"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DOPENSSL_SMALL=1
        ${ExternalProject_CMAKE_ARGS_hidden}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include
    COMMAND ${CMAKE_COMMAND} -E copy <INSTALL_DIR>/src/ext_boringssl-build/ssl/libssl.a <INSTALL_DIR>/lib/libssl.a
    COMMAND ${CMAKE_COMMAND} -E copy <INSTALL_DIR>/src/ext_boringssl-build/crypto/libcrypto.a <INSTALL_DIR>/lib/libcrypto.a
)

#-DCMAKE_BUILD_TYPE=Release 
#-DOPENSSL_SMALL=1
#-DBORINGSSL_DISPATCH_TEST=OFF
# remove_definitions(-Werror=unused-result)

set(BORINGSSL_BUILD_BYPRODUCTS "<INSTALL_DIR>/src/ext_boringssl-build/ssl/libssl.a"
                               "<INSTALL_DIR>/src/ext_boringssl-build/crypto/libcryp.a"
)

ExternalProject_Get_Property(ext_boringssl SOURCE_DIR)
message(STATUS "BORING_SSL_INSTALL_DIR ${INSTALL_DIR}")

ExternalProject_Get_Property(ext_boringssl INSTALL_DIR)
message(STATUS "BORING_SSL_SOURCE_DIR ${SOURCE_DIR}")


set(BORINGSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(BORINGSSL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(BORINGSSL_LIBRARIES ssl crypto)
