include(ExternalProject)

ExternalProject_Add(
    ext_cub
    PREFIX cub
    URL https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.tar.gz
    URL_HASH SHA256=025658f4c933cd2aa8cc88a559d013338d68de3fa639cc1f2b12cf61dc759667
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/cub"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cub SOURCE_DIR)
set(CUB_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
