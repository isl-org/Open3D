include(ExternalProject)

ExternalProject_Add(
    ext_tinygltf
    PREFIX tinygltf
    URL https://github.com/syoyo/tinygltf/archive/72f4a55edd54742bca1a71ade8ac70afca1d3f07.tar.gz
    URL_HASH SHA256=9e848dcf0ec7dcb352ced782aea32064a63a51b3c68ed14c68531e08632a2d90
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tinygltf"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_tinygltf SOURCE_DIR)
set(TINYGLTF_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
