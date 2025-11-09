include(ExternalProject)

ExternalProject_Add(
    ext_cutlass
    PREFIX cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.2.1.tar.gz
    URL_HASH SHA256=a4513ba33ae82fd754843c6d8437bee1ac71a6ef1c74df886de2338e3917d4df
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/cutlass"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cutlass SOURCE_DIR)
set(CUTLASS_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
