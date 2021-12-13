include(ExternalProject)

if(BUILD_WEBRTC)
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
    # 5. googlesource.com's archive does not have the same checksum each time you
    #    download (strange!), so we cannot compare checksum. Instead, we store the
    #    archive in open3d_downloads.
    #    If you want to directly download from googlesource.com, download it from:
    #    https://boringssl.googlesource.com/boringssl/+archive/${boringssl_commit}.tar.gz
    set(boringssl_commit edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd)

    ExternalProject_Add(
        ext_openssl
        PREFIX openssl
        URL "https://github.com/isl-org/open3d_downloads/releases/download/boringssl-tar/edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd.tar.gz"
        URL_HASH SHA256=dba55d212edefed049d523c2711177914a83f849fade23c970101d285c3d4906
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/boringssl"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include
    )

    ExternalProject_Get_Property(ext_openssl INSTALL_DIR)
    set(OPENSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(OPENSSL_LIB_DIR)   # Empty, as we use the sysmbols from WebRTC.
    set(OPENSSL_LIBRARIES) # Empty, as we use the sysmbols from WebRTC.
    set(OPENSSL_ROOT_DIR_FOR_CURL ${INSTALL_DIR})
else()
    # If you have to build BoringSSL, here's how to do it (install golang first):
    #
    # set(libssl_name    ${CMAKE_STATIC_LIBRARY_PREFIX}ssl${CMAKE_STATIC_LIBRARY_SUFFIX})
    # set(libcrypto_name ${CMAKE_STATIC_LIBRARY_PREFIX}crypto${CMAKE_STATIC_LIBRARY_SUFFIX})
    # ExternalProject_Add(
    #     ext_openssl
    #     PREFIX boringssl
    #     URL "https://github.com/isl-org/open3d_downloads/releases/download/boringssl-tar/edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd.tar.gz"
    #     URL_HASH SHA256=dba55d212edefed049d523c2711177914a83f849fade23c970101d285c3d4906
    #     DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/boringssl"
    #     CMAKE_ARGS ${ExternalProject_CMAKE_ARGS_hidden}
    #     BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target ssl crypto
    #     INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include        <INSTALL_DIR>/include
    #     COMMAND         ${CMAKE_COMMAND} -E copy <BINARY_DIR>/ssl/${libssl_name}       <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libssl_name}
    #     COMMAND         ${CMAKE_COMMAND} -E copy <BINARY_DIR>/crypto/${libcrypto_name} <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libcrypto_name}
    #     BUILD_BYPRODUCTS <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libssl_name}
    #                     <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${libcrypto_name}
    # )
    # ExternalProject_Get_Property(ext_openssl INSTALL_DIR)
    # set(OPENSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    # set(OPENSSL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
    # set(OPENSSL_LIBRARIES ssl crypto)
    # set(OPENSSL_ROOT_DIR_FOR_CURL ${INSTALL_DIR})

    # We use OpenSSL as it does not require golang to be installed.
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
        CONFIGURE_COMMAND ./config --prefix=<INSTALL_DIR> --openssldir=<INSTALL_DIR>
        BUILD_COMMAND   make CC="${CMAKE_C_COMPILER}" -j${NPROC}
        INSTALL_COMMAND make CC="${CMAKE_C_COMPILER}" install_sw
    )

    set(OPENSSL_BUILD_BYPRODUCTS "<INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ssl${CMAKE_STATIC_LIBRARY_SUFFIX}"
                                 "<INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}crypto${CMAKE_STATIC_LIBRARY_SUFFIX}"
    )

    ExternalProject_Get_Property(ext_openssl INSTALL_DIR)
    set(OPENSSL_INCLUDE_DIRS ${INSTALL_DIR}/include/ ${INSTALL_DIR}/src/ext_openssl/) # "/" is critical.
    set(OPENSSL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
    set(OPENSSL_LIBRARIES ssl crypto)
    set(OPENSSL_ROOT_DIR_FOR_CURL ${INSTALL_DIR})
endif()

