include(ExternalProject)
include(FetchContent)

# ExternalProject seems to be the best solution for including zeromq.
# The projects defines options which clash with and pollute our CMake cache.

ExternalProject_Add(
    ext_zeromq
    PREFIX "${CMAKE_BINARY_DIR}/zeromq"
    URL "https://github.com/zeromq/libzmq/releases/download/v4.3.2/zeromq-4.3.2.tar.gz"
    URL_HASH MD5=2047e917c2cc93505e2579bcba67a573
    # do not update
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DBUILD_STATIC=ON
        -DBUILD_SHARED=OFF
        -DBUILD_TESTS=OFF
        -DENABLE_CPACK=OFF
        -DENABLE_CURVE=OFF
        -DZMQ_BUILD_TESTS=OFF
        -DWITH_DOCS=OFF
        -DWITH_PERF_TOOL=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

# cppzmq is header only. we just need to download
FetchContent_Declare(
    ext_cppzmq
    URL "https://github.com/zeromq/cppzmq/archive/v4.6.0.tar.gz"
    URL_HASH MD5=7cae1b3fbfeaddb9cf1f70e99a98add2
)
FetchContent_GetProperties(ext_cppzmq)
if(NOT ext_cppzmq_POPULATED)
    FetchContent_Populate(ext_cppzmq)
    # do not add subdirectory here
endif()

ExternalProject_Get_Property( ext_zeromq INSTALL_DIR )
set(ZEROMQ_LIBRARIES zmq)
set(ZEROMQ_LIB_DIR ${INSTALL_DIR}/lib)
set(ZEROMQ_INCLUDE_DIRS "${INSTALL_DIR}/include/;${ext_cppzmq_SOURCE_DIR}/")
