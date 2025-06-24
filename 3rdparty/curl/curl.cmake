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


if (APPLE)  # homebrew does not package libidn2
    set(curl_cmake_extra_args -DUSE_APPLE_IDN=ON -DUSE_LIBIDN2=OFF -DUSE_NGHTTP2=OFF)
endif()

ExternalProject_Add(
    ext_curl
    PREFIX curl
    URL https://github.com/curl/curl/releases/download/curl-8_14_1/curl-8.14.1.tar.xz
    URL_HASH SHA256=f4619a1e2474c4bbfedc88a7c2191209c8334b48fa1f4e53fd584cc12e9120dd
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/curl"
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_CURL_EXE=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_TESTING=OFF
        -DBUILD_LIBCURL_DOCS=OFF
        -DBUILD_MISC_DOCS=OFF
        -DCURL_BROTLI=OFF
        -DCURL_DISABLE_LDAP=ON
        -DCURL_DISABLE_LDAPS=ON
        -DCURL_DISABLE_IMAP=ON
        -DCURL_DISABLE_MQTT=ON
        -DCURL_DISABLE_POP3=ON
        -DCURL_DISABLE_SMTP=ON
        -DCURL_DISABLE_TELNET=ON
        -DCURL_USE_LIBSSH2=OFF
        -DCURL_USE_OPENSSL=ON
        -DCURL_USE_LIBPSL=OFF
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

add_dependencies(ext_curl ext_boringssl)
