# TBB build scripts.
#
# - TBB_INCLUDE_DIR
# - TBB_LIB_DIR
# - TBB_LIBRARIES

include(ExternalProject)

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with mkl.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")

# TBB variables exported for PyTorch Ops and TensorFlow Ops
set(TBB_INCLUDE_DIR "${STATIC_MKL_INCLUDE_DIR}")
set(TBB_LIB_DIR "${STATIC_MKL_LIB_DIR}")
set(TBB_LIBRARIES tbb tbbmalloc)

ExternalProject_Add(
    ext_tbb
    PREFIX tbb
    URL https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.12.0.tar.gz # April 2024
    URL_HASH SHA256=c7bb7aa69c254d91b8f0041a71c5bcc3936acb64408a1719aec0b2b7639dd84f
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tbb"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${MKL_INSTALL_PREFIX}
        -DSTATIC_WINDOWS_RUNTIME=${STATIC_WINDOWS_RUNTIME}
        -DTBBMALLOC_BUILD=ON
        -DTBBMALLOC_PROXY_BUILD=OFF
        -DTBB_TEST=OFF
        -DCMAKE_INSTALL_LIBDIR=${Open3D_INSTALL_LIB_DIR}
        ${ExternalProject_CMAKE_ARGS}
    BUILD_BYPRODUCTS
    ${TBB_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}tbb${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${TBB_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}tbbmalloc${CMAKE_SHARED_LIBRARY_SUFFIX}
)

# TBB is built and linked as a shared library - this is different from all other Open3D dependencies.
add_library(3rdparty_tbb INTERFACE)
target_include_directories(3rdparty_tbb SYSTEM INTERFACE $<BUILD_INTERFACE:${TBB_INCLUDE_DIR}>)
set(TBB_LIBRARIES_PATH "")
foreach(TBB_LIBRARY IN LISTS TBB_LIBRARIES)
    file(REAL_PATH ${CMAKE_SHARED_LIBRARY_PREFIX}${TBB_LIBRARY}${CMAKE_SHARED_LIBRARY_SUFFIX}
        TBB_LIBRARY_PATH BASE_DIRECTORY ${TBB_LIB_DIR})
    list(APPEND TBB_LIBRARIES_PATH ${TBB_LIBRARY_PATH})
endforeach()
target_link_libraries(3rdparty_tbb INTERFACE ${TBB_LIBRARIES_PATH})
add_dependencies(3rdparty_tbb ext_tbb)
add_library(${PROJECT_NAME}::3rdparty_tbb ALIAS 3rdparty_tbb)
install(TARGETS 3rdparty_tbb EXPORT ${PROJECT_NAME}Targets)
install(FILES ${TBB_LIBRARIES_PATH} DESTINATION ${Open3D_INSTALL_LIB_DIR})
