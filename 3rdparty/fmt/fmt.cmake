include(ExternalProject)

set(FMT_LIB_NAME fmt)

ExternalProject_Add(
    ext_fmt
    PREFIX fmt
    URL https://github.com/fmtlib/fmt/archive/refs/tags/6.0.0.tar.gz
    URL_HASH SHA256=f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78
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
