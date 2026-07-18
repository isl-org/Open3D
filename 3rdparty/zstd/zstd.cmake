include(ExternalProject)

if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
            NOT MINGW))
    set(zstd_lib_name zstd_static)
else()
    set(zstd_lib_name zstd)
endif()

ExternalProject_Add(
    ext_zstd
    PREFIX zstd
    URL https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz
    URL_HASH SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/zstd"
    UPDATE_COMMAND ""
    SOURCE_SUBDIR build/cmake
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DZSTD_BUILD_STATIC=ON
        -DZSTD_BUILD_SHARED=OFF
        -DZSTD_BUILD_PROGRAMS=OFF
        -DZSTD_BUILD_TESTS=OFF
        -DZSTD_MULTITHREAD_SUPPORT=OFF
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${zstd_lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_zstd INSTALL_DIR)
set(ZSTD_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(ZSTD_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(ZSTD_LIBRARIES ${zstd_lib_name})
set(ZSTD_INSTALL_DIR ${INSTALL_DIR})
