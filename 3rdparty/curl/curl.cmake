include(ExternalProject)

if(NOT DEFINED BORINGSSL_ROOT_DIR)
    message(FATAL_ERROR "BORINGSSL_ROOT_DIR not set. "
                        "Please include openssl.cmake before including this file.")
endif()

if(MSVC)
    set(curl_lib_name libcurl)
else()
    set(curl_lib_name curl)
endif()

# Ending hosting of new curl binaries to allow easy version updates.
# Keep config comments for future reference.
#
# if(UNIX AND NOT APPLE AND NOT LINUX_AARCH64)
#     # For ubuntu x86, we do not compile from source. Instead, we use the
#     # downloaded libcurl binaries.
#     option(BUILD_CURL_FROM_SOURCE "Build CURL from source" OFF)
# else()
#     option(BUILD_CURL_FROM_SOURCE "Build CURL from source" ON)
# endif()
#
# mark_as_advanced(BUILD_CURL_FROM_SOURCE)
#
# if((MSVC OR APPLE OR LINUX_AARCH64) AND NOT BUILD_CURL_FROM_SOURCE)
#     message(FATAL_ERROR "BUILD_CURL_FROM_SOURCE is required to be ON.")
# endif()

if (APPLE)  # homebrew does not package libidn2
    set(curl_cmake_extra_args -DUSE_APPLE_IDN=ON -DUSE_LIBIDN2=OFF)
endif()

# if(BUILD_CURL_FROM_SOURCE)
ExternalProject_Add(
    ext_curl
    PREFIX curl
    URL https://github.com/curl/curl/releases/download/curl-8_12_1/curl-8.12.1.tar.xz
    URL_HASH SHA256=0341f1ed97a26c811abaebd37d62b833956792b7607ea3f15d001613c76de202
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/curl"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_CURL_EXE=OFF
        -DBUILD_TESTING=OFF
        -DCURL_DISABLE_LDAP=ON
        -DCURL_DISABLE_LDAPS=ON
        -DCURL_DISABLE_IMAP=ON
        -DCURL_DISABLE_MQTT=ON
        -DCURL_DISABLE_POP3=ON
        -DCURL_DISABLE_SMTP=ON
        -DCURL_DISABLE_TELNET=ON
        -DCURL_USE_LIBSSH2=OFF
        -DCURL_USE_OPENSSL=ON
        -DOPENSSL_ROOT_DIR=${BORINGSSL_ROOT_DIR}
        ${curl_cmake_extra_args}
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${curl_lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${curl_lib_name}-d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_curl INSTALL_DIR)
set(CURL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CURL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
if(MSVC)
    set(CURL_LIBRARIES ${curl_lib_name}$<$<CONFIG:Debug>:-d>)
else()
    set(CURL_LIBRARIES ${curl_lib_name})
endif()
    # else()
    #     # Optimize for Ubuntu x86. Curl can take a long time to configure.
    #     #
    #     # To generate pre-compiled curl:
    #     # 1. Use oldest supported Ubuntu (eg. in docker), not the latest.
    #     # 2. -DBUILD_CURL_FROM_SOURCE=ON, build Open3D: make ext_curl
    #     # 3. cd build/curl
    #     # 4. tar -czvf curl_8.10.1_linux_x86_64.tar.gz include lib
    #     ExternalProject_Add(
    #         ext_curl
    #         PREFIX curl
    #         URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/curl_8.10.1_linux_x86_64.tar.bz2
    #         URL_HASH SHA256=4977fc20d00b22ab36e3bebd556f06c720b6c03316cd476487ab37fc79d44163
    #         DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/curl"
    #         UPDATE_COMMAND ""
    #         CONFIGURE_COMMAND ""
    #         BUILD_COMMAND ""
    #         INSTALL_COMMAND ""
    #         BUILD_BYPRODUCTS
    #             <SOURCE_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${curl_lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    #             <SOURCE_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${curl_lib_name}-d${CMAKE_STATIC_LIBRARY_SUFFIX}
    #     )
    #
    #     ExternalProject_Get_Property(ext_curl SOURCE_DIR)
    #     set(CURL_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
    #     set(CURL_LIB_DIR ${SOURCE_DIR}/lib)
    #     set(CURL_LIBRARIES ${curl_lib_name})
    # endif()

add_dependencies(ext_curl ext_boringssl)