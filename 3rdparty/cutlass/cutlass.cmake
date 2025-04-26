include(ExternalProject)

ExternalProject_Add(
    ext_cutlass
    PREFIX cutlass
    URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.9.0.tar.gz
    URL_HASH SHA256=0ea98a598d1f77fade5187ff6ec6d9e6ef3acd267ee68850aae6e800dcbd69c7
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/cutlass"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cutlass SOURCE_DIR)
set(CUTLASS_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
