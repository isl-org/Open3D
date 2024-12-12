# MKL and TBB build scripts.
#
# This scripts exports: (both MKL and TBB)
# - STATIC_MKL_INCLUDE_DIR
# - STATIC_MKL_LIB_DIR
# - STATIC_MKL_LIBRARIES
#
# The name "STATIC" is used to avoid naming collisions for other 3rdparty CMake
# files (e.g. PyTorch) that also depends on MKL.

include(ExternalProject)

# These files are created from the pip MKL devel packages, and only contain
# headers, static libraries, and cmake export files. Shared libraries are
# excluded to reduce download size. Alternately, use:
# pip download -d mkl_static/win_amd64 --platform win_amd64 --no-deps mkl-include==2024.1 mkl-devel==2024.1 mkl-static==2024.1
# pip download -d mkl_static/linux_x86_64 --platform manylinux1_x86_64 --no-deps mkl-include==2024.1 mkl-devel==2024.1 mkl-static==2024.1
# pip download -d mkl_static/macosx_x86_64 --platform macosx_11_0_x86_64 --no-deps mkl-include==2023.2.2 mkl-devel==2023.2.2 mkl-static==2023.2.2
# Extract all files:
# cd mkl_static/win_amd64 && for whl in *.whl; do wheel unpack $whl; done;
# Arrange in the standard layout: bin, include, lib (cmake, pkgconfig), share (cmake)
# Archive and upload to GitHub releases open3d_downloads.
if(WIN32)
    set(MKL_URL https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/mkl_static-2024.1.0-win_amd64.zip)
    set(MKL_SHA256 524de5395db5b7a9d9f0d9a76b2223c6edac429d4492c6a1cc79a5c22c4f3346)
elseif(APPLE)
    set(MKL_URL https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/mkl_static-2023.2.2.9-macosx_x86_64.tar.xz)
    set(MKL_SHA256 6cd93bf1d37527d3ab3657e22c1a8a409729d6c6f422c7c381c7a145aa588d6c)
else()
    set(MKL_URL https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/mkl_static-2024.1.0-linux_x86_64.tar.xz)
    set(MKL_SHA256 f37c9440e3d664d21889a4607effcd47472bcce347da6c2bfc7aae991971b499)
endif()

# Where MKL and TBB headers and libs will be installed.
# This needs to be consistent with tbb.cmake.
set(MKL_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/mkl_install)
set(STATIC_MKL_INCLUDE_DIR "${MKL_INSTALL_PREFIX}/include/")
set(STATIC_MKL_LIB_DIR "${MKL_INSTALL_PREFIX}/${Open3D_INSTALL_LIB_DIR}")

if(WIN32)
    ExternalProject_Add(
        ext_mkl
        PREFIX mkl
        URL ${MKL_URL}
        URL_HASH SHA256=${MKL_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Library/lib ${STATIC_MKL_LIB_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/Library/include ${MKL_INSTALL_PREFIX}/include
        BUILD_BYPRODUCTS
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_intel_ilp64${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_core${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_sequential${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_tbb_thread${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    # Generator expression can result in an empty string "", causing CMake to try to
    # locate ".lib". The workaround to first list all libs, and remove unneeded items
    # using generator expressions.
    set(STATIC_MKL_LIBRARIES
        mkl_intel_ilp64
        mkl_core
        mkl_sequential
        mkl_tbb_thread
    )
    list(REMOVE_ITEM STATIC_MKL_LIBRARIES "$<$<CONFIG:Debug>:mkl_tbb_thread>")
    list(REMOVE_ITEM STATIC_MKL_LIBRARIES "$<$<CONFIG:Release>:mkl_sequential>")
else()
    ExternalProject_Add(
        ext_mkl
        PREFIX mkl
        URL ${MKL_URL}
        URL_HASH SHA256=${MKL_SHA256}
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/mkl"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/lib ${STATIC_MKL_LIB_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${MKL_INSTALL_PREFIX}/include
        BUILD_BYPRODUCTS
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_intel_ilp64${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_tbb_thread${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${STATIC_MKL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}mkl_core${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    set(STATIC_MKL_LIBRARIES mkl_intel_ilp64 mkl_tbb_thread mkl_core)
endif()
