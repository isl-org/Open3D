include(ExternalProject)

ExternalProject_Add(
    ext_qhull
    PREFIX qhull
    URL https://github.com/qhull/qhull/archive/refs/tags/v7.3.0.tar.gz
    URL_HASH SHA256=05a2311d8e6397c96802ee5a9d8db32b83dac7e42e2eb2cd81c5547c18e87de6
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/qhull"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_qhull SOURCE_DIR)
set(QHULL_SOURCE_DIR ${SOURCE_DIR})
