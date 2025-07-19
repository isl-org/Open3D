include(ExternalProject)

if(MSVC)
    set(LIBPNG_LIB_NAME libpng16_static)
else()
    set(LIBPNG_LIB_NAME png16)
endif()

ExternalProject_Add(
    ext_libpng
    PREFIX libpng
    URL https://github.com/pnggroup/libpng/archive/refs/tags/v1.6.49.tar.gz
    URL_HASH SHA256=e425762fdfb9bb30a5d2da29c0067570e96b5d41d79c659cf0dad861e9df738e
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/libpng"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DPNG_SHARED=OFF
        -DPNG_TOOLS=OFF
        -DPNG_TESTS=OFF
        -DZLIB_ROOT=${CMAKE_BINARY_DIR}/zlib
        -DPNG_ARM_NEON=off # Must be lower case.
        ${ExternalProject_CMAKE_ARGS_hidden}
    DEPENDS ext_zlib
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${LIBPNG_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${LIBPNG_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_libpng INSTALL_DIR)
set(LIBPNG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(LIBPNG_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(LIBPNG_LIBRARIES ${LIBPNG_LIB_NAME}$<$<PLATFORM_ID:Windows>:$<$<CONFIG:Debug>:d>>)
