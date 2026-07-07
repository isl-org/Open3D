include(ExternalProject)

ExternalProject_Add(
    ext_civetweb
    PREFIX civetweb
    URL https://github.com/civetweb/civetweb/archive/refs/tags/v1.16.tar.gz
    URL_HASH SHA256=f0e471c1bf4e7804a6cfb41ea9d13e7d623b2bcc7bc1e2a4dd54951a24d60285
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/civetweb"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCIVETWEB_BUILD_TESTING=OFF
        -DCIVETWEB_ENABLE_CXX=ON
        -DCIVETWEB_SSL_OPENSSL_API_1_0=OFF
        -DCIVETWEB_SSL_OPENSSL_API_1_1=ON
        -DCIVETWEB_ENABLE_SERVER_EXECUTABLE=OFF
        -DCIVETWEB_ENABLE_ASAN=OFF
        -DCIVETWEB_ENABLE_DEBUG_TOOLS=OFF
        -DCIVETWEB_DISABLE_CGI=ON
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        # Note: on Windows, keep this as a single flag (no embedded space by
        # combining with another flag in the same string). The Intel
        # oneAPI DPC++/C++ compiler (icx), used when BUILD_SYCL_MODULE=ON,
        # parses each CMAKE_ARGS string as one literal argv token; a
        # combined "-D<macro> /EHsc" string gets misparsed by icx as the
        # macro's value swallowing "/EHsc", silently leaving exceptions
        # disabled (MSVC cl.exe tolerates this, icx does not). Since
        # _GLIBCXX_USE_CXX11_ABI only matters for libstdc++ (not used on
        # Windows/MSVC STL), it is safe to only set one flag per platform.
        -DCMAKE_CXX_FLAGS=$<IF:$<PLATFORM_ID:Windows>,/EHsc,-D_GLIBCXX_USE_CXX11_ABI=$<BOOL:${GLIBCXX_USE_CXX11_ABI}>>
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}civetweb${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}civetweb-cpp${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_civetweb INSTALL_DIR)
set(CIVETWEB_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CIVETWEB_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(CIVETWEB_LIBRARIES civetweb civetweb-cpp)
