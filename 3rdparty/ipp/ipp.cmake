# Output variables:
# - IPP_INCLUDE_DIR
# - IPP_LIBRARIES
# - IPP_LIB_DIR

include(ExternalProject)

# Check in order APPLE -> WIN32 -> UNIX, since UNIX may be defined on APPLE / WIN32 as well
set(IPP_VERSION_STRING "2021.11.0")  # From ipp/ippversion.h
if(APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(IPP_VERSION_STRING "2021.9.1")  # From ipp/ippversion.h
    set(IPP_URL "https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/ipp_static-2021.9.1-macosx_10_15_x86_64.tar.xz")
    set(IPP_HASH "f27e45da604a1f6d1d2a747a0f67ffafeaff084b0f860a963d8c3996e2f40bb3")
elseif(WIN32 AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL AMD64)
    set(IPP_URL "https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/ipp_static-2021.11.0-win_amd64.zip") 
    set(IPP_HASH "69e8a7dc891609de6fea478a67659d2f874d12b51a47bd2e3e5a7c4c473c53a6")
elseif(UNIX AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL x86_64)
    set(IPP_URL "https://github.com/isl-org/open3d_downloads/releases/download/mkl-static-2024.1/ipp_static-2021.11.0-linux_x86_64.tar.xz")
    set(IPP_HASH "51f33fd5bf5011e9eae0e034e5cc70a7c0ac0ba93d6a3f66fd7e145cf1a5e30b")
else()
    set(WITH_IPP OFF)
    message(FATAL_ERROR "Intel IPP disabled: Unsupported Platform.")
    return()
endif()

if(WIN32)
    set(TL mt_tl_tbb)
    set(IPP_SUBPATH "Library/")
else()
    set(TL _tl_tbb)
    set(IPP_SUBPATH "")
endif()

ExternalProject_Add(ext_ipp
    PREFIX ipp
    URL ${IPP_URL} 
    URL_HASH SHA256=${IPP_HASH}
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/ipp"
    # Copy all libs from lib/tl/tbb to lib/ since Open3D cmake scripts only support one LIB_DIR per dependency
    UPDATE_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/${IPP_SUBPATH}lib/tl/tbb/ <SOURCE_DIR>/${IPP_SUBPATH}lib/
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
ExternalProject_Get_Property(ext_ipp SOURCE_DIR)
set(IPP_INCLUDE_DIR "${SOURCE_DIR}/${IPP_SUBPATH}include/")
# Threading layer libs must be linked first.
# https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2021-11/ipp-performace-benefits-with-tl-functions.html
# Library dependency order:
# https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2021-11/library-dependencies-by-domain.html
set(IPP_LIBRARIES ipp_iw ippcv${TL} ippcv ippi${TL} ippi ippcc${TL} ippcc ipps ippvm ippcore${TL} ippcore)
set(IPP_LIB_DIR "${SOURCE_DIR}/${IPP_SUBPATH}lib")