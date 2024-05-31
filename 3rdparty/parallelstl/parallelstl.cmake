include(ExternalProject)

ExternalProject_Add(
    ext_parallelstl
    PREFIX parallelstl
    URL https://github.com/oneapi-src/oneDPL/archive/refs/tags/oneDPL-2022.5.0-rc1.tar.gz
    URL_HASH SHA256=9180c60331ec5b307dd89a5d8bfcd096667985c6761c52322405d4b69193ed88
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/parallelstl"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_parallelstl SOURCE_DIR)
set(PARALLELSTL_INCLUDE_DIRS ${SOURCE_DIR}/include/) # "/" is critical.
