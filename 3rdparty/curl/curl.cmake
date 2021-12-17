include(ExternalProject)
include(ProcessorCount)
ProcessorCount(NPROC)

if(NOT DEFINED OPENSSL_ROOT_DIR)
    message(FATAL_ERROR "OPENSSL_ROOT_DIR not set. "
                        "Please include openssl.cmake before including this file.")
endif()

# TODO (rishabh):
# - determine "-DCMAKE_USE_OPENSSL=ON"
# - try windows Debug/Release, static/dynamic runtime, as long as ci passing
#   otherwise, we'll need different binaries for each case
#   CMAKE_MSVC_RUNTIME_LIBRARY and build type for Windows
if(MSVC)
    set(curl_lib_name libcurl)
    set(curl_cmake_extra_args "-DCMAKE_USE_OPENSSL=ON")
else()
    set(curl_lib_name curl)
endif()

# TODO: for MSVC https://stackoverflow.com/a/30011890/1255535
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
        -DCURL_USE_LIBSSH2=OFF
        -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}
        ${curl_cmake_extra_args}
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${curl_lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
    BUILD_ALWAYS ON
)
add_dependencies(ext_curl ext_openssl)
ExternalProject_Get_Property(ext_curl INSTALL_DIR)
set(CURL_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CURL_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
if(MSVC)
    set(CURL_LIBRARIES ${curl_lib_name}$<$<CONFIG:Debug>:-d>)
else()
    set(CURL_LIBRARIES ${curl_lib_name})
endif()

