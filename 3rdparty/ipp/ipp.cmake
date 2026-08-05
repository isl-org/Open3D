# Output variables:
# - IPP_INCLUDE_DIR
# - IPP_LIBRARIES
# - IPP_LIB_DIR
# - IPP_VERSION_STRING
# - IPP_VERSION_INT    (for version check)

include(ExternalProject)

# These archives are created from the pip wheels for ipp-static, ipp-devel and
# ipp-include and excluding the shared libraries to reduce download size.
# Check in order APPLE -> WIN32 -> UNIX, since UNIX may be defined on APPLE / WIN32 as well
set(IPP_VERSION_STRING "2021.11.0")  # From ipp/ippversion.h
set(IPP_VERSION_INT 20211100)
if(APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(IPP_VERSION_STRING "2021.9.1")  # From ipp/ippversion.h
    set(IPP_VERSION_INT 20210901)
    set(IPP_URL "https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/ipp_static-2021.9.1-macosx_10_15_x86_64.tar.xz")
    set(IPP_HASH "f27e45da604a1f6d1d2a747a0f67ffafeaff084b0f860a963d8c3996e2f40bb3")
    set(COPY_TBB_COMMAND cp -rp)
elseif(WIN32 AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL AMD64)
    set(IPP_URL "https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/ipp_static-2021.11.0-win_amd64.zip") 
    set(IPP_HASH "69e8a7dc891609de6fea478a67659d2f874d12b51a47bd2e3e5a7c4c473c53a6")
    cmake_minimum_required(VERSION 3.26)  # for copy_directory_if_different
    set(COPY_TBB_COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different)
elseif(UNIX AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(IPP_URL "https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/ipp_static-2021.11.0-linux_x86_64.tar.xz")
    set(IPP_HASH "51f33fd5bf5011e9eae0e034e5cc70a7c0ac0ba93d6a3f66fd7e145cf1a5e30b")
    set(COPY_TBB_COMMAND cp -rp)
else()
    set(WITH_IPP OFF)
    message(FATAL_ERROR "Intel IPP disabled: Unsupported Platform.")
    return()
endif()

if(WIN32)
    set(IPP_SUBPATH "Library/")
else()
    set(IPP_SUBPATH "")
endif()

# Threading layer libs must be linked first.
# https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2021-11/ipp-performace-benefits-with-tl-functions.html
# Library dependency order:
# https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2021-11/library-dependencies-by-domain.html
if (WIN32)
    set(IPP_LIBRARIES ipp_iw ippcvmt_tl_tbb ippcvmt ippimt_tl_tbb ippimt ippccmt_tl_tbb ippccmt ippsmt ippvmmt ippcoremt_tl_tbb ippcoremt)
else()
    set(IPP_LIBRARIES ipp_iw ippcv_tl_tbb ippcv ippi_tl_tbb ippi ippcc_tl_tbb ippcc ipps ippvm ippcore_tl_tbb ippcore)
endif()

foreach(item IN LISTS IPP_LIBRARIES)
    list(APPEND IPP_BUILD_BYPRODUCTS <SOURCE_DIR>/${IPP_SUBPATH}lib/${CMAKE_STATIC_LIBRARY_PREFIX}${item}${CMAKE_STATIC_LIBRARY_SUFFIX})
endforeach()

ExternalProject_Add(ext_ipp
    PREFIX ipp
    URL ${IPP_URL} 
    URL_HASH SHA256=${IPP_HASH}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/ipp"
    # Copy all libs from lib/tl/tbb to lib/ since Open3D cmake scripts only support one LIB_DIR per dependency
    UPDATE_COMMAND ${COPY_TBB_COMMAND} <SOURCE_DIR>/${IPP_SUBPATH}lib/tl/tbb/. <SOURCE_DIR>/${IPP_SUBPATH}lib/
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
        ${IPP_BUILD_BYPRODUCTS}
)

ExternalProject_Get_Property(ext_ipp SOURCE_DIR)
set(IPP_INCLUDE_DIR "${SOURCE_DIR}/${IPP_SUBPATH}include/")
set(IPP_LIB_DIR "${SOURCE_DIR}/${IPP_SUBPATH}lib")
