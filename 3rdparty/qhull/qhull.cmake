include(ExternalProject)

ExternalProject_Add(
    ext_qhull
    PREFIX qhull
    # v8.0.0+ causes seg fault
    URL https://github.com/qhull/qhull/archive/refs/tags/v8.0.2.tar.gz
    URL_HASH
    SHA256=8774e9a12c70b0180b95d6b0b563c5aa4bea8d5960c15e18ae3b6d2521d64f8b
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/qhull"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_qhull SOURCE_DIR)
set(QHULL_SOURCE_DIR ${SOURCE_DIR})
