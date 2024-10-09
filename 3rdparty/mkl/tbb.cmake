# TBB build scripts.

include(FetchContent)
cmake_policy(SET CMP0077 NEW)

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with mkl.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_INCLUDE_DIR}/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")

# Save and restore BUILD_SHARED_LIBS since TBB must be built as a shared library
set(_build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS ON)
set(_win_exp_all_syms ${CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS})
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)   # ON interferes with TBB symbols
FetchContent_Declare(
    ext_tbb
    URL https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.12.0.tar.gz # April 2024
    URL_HASH SHA256=c7bb7aa69c254d91b8f0041a71c5bcc3936acb64408a1719aec0b2b7639dd84f
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tbb"
)
set(TBBMALLOC_BUILD OFF CACHE BOOL "Enable tbbmalloc build.")
set(TBBMALLOC_PROXY_BUILD OFF CACHE BOOL "Enable tbbmalloc_proxy build.")
set(TBB_TEST OFF CACHE BOOL "Build TBB tests.")
set(TBB_INSTALL OFF CACHE BOOL "Enable installation")
set(TBB_STRICT OFF CACHE BOOL "Treat compiler warnings as errors")
FetchContent_MakeAvailable(ext_tbb)
set(BUILD_SHARED_LIBS ${_build_shared_libs})
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ${_win_exp_all_syms})

# TBB is built and linked as a shared library - this is different from all other Open3D dependencies.
install(TARGETS tbb EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}     # Windows .lib files
  COMPONENT tbb
  LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
  COMPONENT tbb
  RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
  COMPONENT tbb
)
add_library(${PROJECT_NAME}::3rdparty_tbb ALIAS tbb)
