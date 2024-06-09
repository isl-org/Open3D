# TBB build scripts.

include(FetchContent)
cmake_policy(SET CMP0077 NEW)

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with mkl.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_INCLUDE_DIR}/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")

# TBB variables exported for PyTorch Ops and TensorFlow Ops
set(TBB_INCLUDE_DIR "${STATIC_MKL_INCLUDE_DIR}")
set(TBB_LIB_DIR "${STATIC_MKL_LIB_DIR}")
set(TBB_RUNTIME_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_BIN_DIR}")
set(TBB_LIBRARIES tbb tbbmalloc)

FetchContent_Declare(
    3rdparty_tbb
    URL https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.12.0.tar.gz # April 2024
    URL_HASH SHA256=c7bb7aa69c254d91b8f0041a71c5bcc3936acb64408a1719aec0b2b7639dd84f
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tbb"
)
set(TBBMALLOC_BUILD ON CACHE BOOL "Build TBB malloc library.")
set(TBBMALLOC_PROXY_BUILD OFF CACHE BOOL "Build TBB malloc proxy library.")
set(TBB_TEST OFF CACHE BOOL "Build TBB tests.")
FetchContent_MakeAvailable(3rdparty_tbb)

# TBB is built and linked as a shared library - this is different from all other Open3D dependencies.
install(TARGETS 3rdparty_tbb EXPORT ${PROJECT_NAME}Targets
LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR})
add_library(${PROJECT_NAME}::3rdparty_tbb ALIAS 3rdparty_tbb)
