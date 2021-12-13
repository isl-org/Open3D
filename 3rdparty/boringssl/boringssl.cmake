include(ExternalProject)

set(libssl_name    ${CMAKE_STATIC_LIBRARY_PREFIX}ssl${CMAKE_STATIC_LIBRARY_SUFFIX})
set(libcrypto_name ${CMAKE_STATIC_LIBRARY_PREFIX}crypto${CMAKE_STATIC_LIBRARY_SUFFIX})

# boringssl_commit should be consistent with the boringssl used by WEBRTC_COMMIT
# in 3rdparty/webrtc/webrtc_build.sh
#
# 1. Clone the webrtc repo, as described in 3rdparty/webrtc/README.md
# 2. Checkout the `WEBRTC_COMMIT`
# 3. Run `cd src/third_party` and `git log --name-status -10 boringssl`
# 4. Read and understand the commit log. The commit in the boringssl repo is not
#    the same as the one used in `third_party`, as the `third_party` repo copies
#    the boringssl code and does not use a submodule. Determine the commit id
#    of boringssl manually.
set(boringssl_commit edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd)

ExternalProject_Add(
    ext_boringssl
    PREFIX boringssl
    URL "https://boringssl.googlesource.com/boringssl/+archive/${boringssl_commit}.tar.gz"
    URL_HASH SHA256=96d0f60f763515d96f6a56fd6f7ea2707bd1965b2a0aea0875efe8591107b7a7
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
