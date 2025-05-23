include(ExternalProject)

# ------------------------------------------------------------------
# Library name (needed by the rest of Open3D build files)
# ------------------------------------------------------------------
if(MSVC)
    set(ZLIB_LIB_NAME zlibstatic)  # .lib produced by zlib‑ng –compat
else()
    set(ZLIB_LIB_NAME z)           # libz.a produced by zlib‑ng –compat
endif()

find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ext_zlib
    PREFIX zlib-ng
    URL https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.2.2.tar.gz
    URL_HASH SHA256=fcb41dd59a3f17002aeb1bb21f04696c9b721404890bb945c5ab39d2cb69654c
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/zlib"
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DZLIB_COMPAT:BOOL=ON
        -DZLIB_ENABLE_TESTS:BOOL=OFF
        -DZLIBNG_ENABLE_TESTS:BOOL=OFF
        -DWITH_GTEST:BOOL=OFF
        # zlib needs visiible symbols for examples. Disabling example building causes
        # assember error in GPU CI. zlib symbols are hidden during linking.
        ${ExternalProject_CMAKE_ARGS}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_zlib INSTALL_DIR)
set(ZLIB_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(ZLIB_LIB_DIR ${INSTALL_DIR}/lib)
if(MSVC)
    set(ZLIB_LIBRARIES ${ZLIB_LIB_NAME}$<$<CONFIG:Debug>:d>)
else()
    set(ZLIB_LIBRARIES ${ZLIB_LIB_NAME})
endif()