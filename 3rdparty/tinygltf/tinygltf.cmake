include(ExternalProject)

ExternalProject_Add(
    ext_tinygltf
    PREFIX tinygltf
    URL https://github.com/syoyo/tinygltf/archive/refs/tags/v2.8.3.tar.gz
    URL_HASH SHA256=fbef83ef47dbc6d1662103b54ea54f05a753ddff7a11d80b9fe0cd306ab5d4d2
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/tinygltf"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_tinygltf SOURCE_DIR)
set(TINYGLTF_INCLUDE_DIRS ${SOURCE_DIR}/) # "/" is critical.
