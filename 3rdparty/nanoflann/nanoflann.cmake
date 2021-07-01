include(ExternalProject)

ExternalProject_Add(
    ext_nanoflann
    PREFIX nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.3.2.tar.gz
    URL_HASH SHA256=e100b5fc8d72e9426a80312d852a62c05ddefd23f17cbb22ccd8b458b11d0bea
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/nanoflann"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_nanoflann SOURCE_DIR)
set(NANOFLANN_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
