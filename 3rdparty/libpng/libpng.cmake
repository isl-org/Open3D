include(ExternalProject)

if(MSVC)
    set(LIBPNG_LIB_NAME libpng16_static)
else()
    set(LIBPNG_LIB_NAME png16)
endif()

ExternalProject_Get_Property(ext_zlib INSTALL_DIR)
set(ZLIB_INSTALL_DIR ${INSTALL_DIR})
unset(INSTALL_DIR)

ExternalProject_Add(
    ext_libpng
    PREFIX libpng
    URL https://github.com/glennrp/libpng/archive/refs/tags/v1.6.47.tar.gz
    URL_HASH SHA256=631a4c58ea6c10c81f160c4b21fa8495b715d251698ebc2552077e8450f30454
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/libpng"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DPNG_SHARED=OFF
        -DPNG_EXECUTABLES=OFF
        -DPNG_TESTS=OFF
        -DPNG_BUILD_ZLIB=OFF
        -DZLIB_ROOT:PATH=${ZLIB_INSTALL_DIR}
        -DPNG_ARM_NEON=off # Must be lower case.
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${LIBPNG_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${LIBPNG_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_libpng INSTALL_DIR)
set(LIBPNG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(LIBPNG_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(LIBPNG_LIBRARIES ${LIBPNG_LIB_NAME}$<$<PLATFORM_ID:Windows>:$<$<CONFIG:Debug>:d>>)
