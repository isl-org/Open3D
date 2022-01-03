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

if(UNIX AND NOT APPLE AND NOT LINUX_AARCH64)
    # For ubuntu x86, we do not compile from source. Instead, we use the
    # downloaded libcurl binaries.
    option(BUILD_CURL_FROM_SOURCE "Build CURL from source" OFF)
else()
    option(BUILD_CURL_FROM_SOURCE "Build CURL from source" ON)
endif()

if(BUILD_CURL_FROM_SOURCE)
    ExternalProject_Add(
        ext_curl
        PREFIX curl
        URL https://github.com/curl/curl/releases/download/curl-7_79_1/curl-7.79.1.tar.gz
        URL_HASH SHA256=370b11201349816287fb0ccc995e420277fbfcaf76206e309b3f60f0eda090c2
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/curl"
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DBUILD_SHARED_LIBS=OFF
            -DBUILD_CURL_EXE=OFF
            -DBUILD_TESTING=OFF
            -DCURL_DISABLE_LDAP=ON
            -DCURL_DISABLE_LDAPS=ON
            -DCMAKE_USE_LIBSSH2=OFF
            -DCMAKE_USE_OPENSSL=ON
            -DOPENSSL_ROOT_DIR=${BORINGSSL_ROOT_DIR}
            ${curl_cmake_extra_args}
            ${ExternalProject_CMAKE_ARGS_hidden}
        BUILD_BYPRODUCTS
            <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${curl_lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Get_Property(ext_curl INSTALL_DIR)
    set(CURL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
    set(CURL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
    if(MSVC)
        set(CURL_LIBRARIES ${curl_lib_name}$<$<CONFIG:Debug>:-d>)
    else()
        set(CURL_LIBRARIES ${curl_lib_name})
    endif()
else()
    # Optimize for Ubuntu x86. Curl can take a long time to configure.
    #
    # To generate pre-compiled curl:
    # 1. Use Ubuntu 18.04, not 20.04+.
    # 2. -DBUILD_CURL_FROM_SOURCE=ON, build Open3D
    # 3. cd build/curl
    # 4. tar -czvf curl_7.79.1_linux_x86_64.tar.gz include lib
    ExternalProject_Add(
        ext_curl
        PREFIX curl
        URL https://github.com/isl-org/open3d_downloads/releases/download/boringssl-bin/curl_7.79.1_linux_x86_64.tar.gz
        URL_HASH SHA256=8ce6b39cea77b6147464f11aa8050f310db6614b23f01a71bc1df3a38633398c
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/curl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ""
    )

    ExternalProject_Get_Property(ext_curl SOURCE_DIR)
    set(CURL_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
    set(CURL_LIB_DIR ${SOURCE_DIR}/lib)
    set(CURL_LIBRARIES ${curl_lib_name})
endif()

add_dependencies(ext_curl ext_boringssl)
