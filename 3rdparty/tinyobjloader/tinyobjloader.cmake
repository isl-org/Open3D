include(ExternalProject)

ExternalProject_Add(
    ext_tinyobjloader
    PREFIX tinyobjloader
    URL https://github.com/tinyobjloader/tinyobjloader/archive/refs/tags/v2.0.0rc8.tar.gz
    URL_HASH SHA256=b8c972dfbbcef33d55554e7c9031abe7040795b67778ad3660a50afa7df6ec56
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tinyobjloader"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_tinyobjloader SOURCE_DIR)
set(TINYOBJLOADER_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
