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

if(WIN32)
    set(MKL_URL
        https://github.com/isl-org/Open3D/releases/download/v0.12.0/mkl-static-2020.1-intel_216-win-64.tar.bz2
        https://anaconda.org/intel/mkl-static/2020.1/download/win-64/mkl-static-2020.1-intel_216.tar.bz2
    )
    set(MKL_SHA256 c6f037aa9e53501d91d5245b6e65020399ebf34174cc4d03637818ebb6e6b6b9)
elseif(APPLE)
    set(MKL_URL /Users/ssheorey/Downloads/mkl_static-2023.2.2.9-macosx_x86_64.tar.xz)
    set(MKL_SHA256 6cd93bf1d37527d3ab3657e22c1a8a409729d6c6f422c7c381c7a145aa588d6c)
else()
    set(MKL_URL /home/ssheorey/mkl_static-2024.1.0-linux_x86_64.tar.xz)
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
        # tbb_static
    )
    list(REMOVE_ITEM MKL_LIBRARIES "$<$<CONFIG:Debug>:mkl_tbb_thread>")
    list(REMOVE_ITEM MKL_LIBRARIES "$<$<CONFIG:Debug>:tbb_static>")
    list(REMOVE_ITEM MKL_LIBRARIES "$<$<CONFIG:Release>:mkl_sequential>")
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
    # set(STATIC_MKL_LIBRARIES mkl_intel_ilp64 mkl_tbb_thread mkl_core tbb_static)
    set(STATIC_MKL_LIBRARIES mkl_intel_ilp64 mkl_tbb_thread mkl_core)
endif()
