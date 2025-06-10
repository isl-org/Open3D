include(ExternalProject)

ExternalProject_Add(
        ext_cutlass
        PREFIX cutlass
        URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.9.2.tar.gz
        URL_HASH SHA256=4b97bd6cece9701664eec3a634a1f2f2061d85bf76d843fa5799e1a692b4db0d
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/cutlass"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cutlass SOURCE_DIR)
set(CUTLASS_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.