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

add_dependencies(ext_curl ext_boringssl)
