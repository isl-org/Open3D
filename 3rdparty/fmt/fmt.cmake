include(ExternalProject)

set(FMT_LIB_NAME fmt)

if (MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # MSVC has errors when building fmt >6, up till 9.1
    # SYCL / DPC++ needs fmt ver <=6 or >= 9.2: https://github.com/fmtlib/fmt/issues/3005
    set(FMT_VER "6.0.0")
    set(FMT_SHA256
        "f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78")
else()
    set(FMT_VER "9.0.0")
    set(FMT_SHA256
        "9a1e0e9e843a356d65c7604e2c8bf9402b50fe294c355de0095ebd42fb9bd2c5")
endif()

ExternalProject_Add(
    ext_fmt
    PREFIX fmt
    URL https://github.com/fmtlib/fmt/archive/refs/tags/${FMT_VER}.tar.gz
    URL_HASH SHA256=${FMT_SHA256}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/fmt"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DFMT_DOC=OFF
        -DFMT_TEST=OFF
        -DFMT_FUZZ=OFF
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${FMT_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${FMT_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_fmt INSTALL_DIR)
set(FMT_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(FMT_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(FMT_LIBRARIES ${FMT_LIB_NAME}$<$<PLATFORM_ID:Windows>:$<$<CONFIG:Debug>:d>>)
