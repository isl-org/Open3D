# TBB build scripts.
#
# - STATIC_TBB_INCLUDE_DIR
# - STATIC_TBB_LIB_DIR
# - STATIC_TBB_LIBRARIES
#
# Notes:
# The name "STATIC" is used to avoid naming collisions for other 3rdparty CMake
# files (e.g. PyTorch) that also depends on MKL.

include(ExternalProject)

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with mkl.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")

# TBB variables exported for PyTorch Ops and TensorFlow Ops
set(STATIC_TBB_INCLUDE_DIR "${STATIC_MKL_INCLUDE_DIR}")
set(STATIC_TBB_LIB_DIR "${STATIC_MKL_LIB_DIR}")
set(STATIC_TBB_LIBRARIES tbb_static tbbmalloc_static)

find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ext_tbb
    PREFIX tbb
    URL https://github.com/wjakob/tbb/archive/141b0e310e1fb552bdca887542c9c1a8544d6503.tar.gz # Sept 2020
    URL_HASH SHA256=bb29b76eabf7549660e3dba2feb86ab501469432a15fb0bf2c21e24d6fbc4c72
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tbb"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${GIT_EXECUTABLE} init
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
        ${CMAKE_CURRENT_LIST_DIR}/0001-Allow-selecttion-of-static-dynamic-MSVC-runtime.patch
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${MKL_INSTALL_PREFIX}
        -DSTATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
        -DTBB_BUILD_TBBMALLOC=ON
        -DTBB_BUILD_TBBMALLOC_PROXYC=OFF
        -DTBB_BUILD_SHARED=OFF
        -DTBB_BUILD_STATIC=ON
        -DTBB_BUILD_TESTS=OFF
        -DTBB_INSTALL_ARCHIVE_DIR=${Open3D_INSTALL_LIB_DIR}
        -DTBB_CMAKE_PACKAGE_INSTALL_DIR=${Open3D_INSTALL_LIB_DIR}/cmake/tbb
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        ${STATIC_TBB_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tbb_static${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${STATIC_TBB_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}tbbmalloc_static${CMAKE_STATIC_LIBRARY_SUFFIX}
)
