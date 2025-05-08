include(ExternalProject)

set(FMT_LIB_NAME fmt)

if (MSVC AND BUILD_CUDA_MODULE)
    if (MSVC_VERSION GREATER_EQUAL 1930)  # v143
        set(FMT_VER "10.1.1")
        set(FMT_SHA256
            "78b8c0a72b1c35e4443a7e308df52498252d1cefc2b08c9a97bc9ee6cfe61f8b")
    else()
        set(FMT_VER "6.0.0")
        set(FMT_SHA256
            "f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78")
    endif()
else()
    set(FMT_VER "10.2.1")
    set(FMT_SHA256
        "1250e4cc58bf06ee631567523f48848dc4596133e163f02615c97f78bab6c811")
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
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${FMT_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${FMT_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_fmt INSTALL_DIR)
set(FMT_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(FMT_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(FMT_LIBRARIES ${FMT_LIB_NAME}$<$<PLATFORM_ID:Windows>:$<$<CONFIG:Debug>:d>>)
