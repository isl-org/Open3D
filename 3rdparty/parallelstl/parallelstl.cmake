include(ExternalProject)

ExternalProject_Add(
    ext_parallelstl
    PREFIX parallelstl
    URL https://github.com/oneapi-src/oneDPL/archive/refs/tags/20190522.tar.gz
    URL_HASH SHA256=40d78c3405a42f781348b5bc9038cb0ce1147591e07fca7329538c9842d36a7b
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/parallelstl"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_parallelstl SOURCE_DIR)
set(PARALLELSTL_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
