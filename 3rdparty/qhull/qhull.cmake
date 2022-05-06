include(ExternalProject)

ExternalProject_Add(
    ext_qhull
    PREFIX qhull
    # v8.0.0+ causes seg fault
    URL https://github.com/qhull/qhull/archive/refs/tags/v7.3.2.tar.gz
    URL_HASH SHA256=619c8a954880d545194bc03359404ef36a1abd2dde03678089459757fd790cb0
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/qhull"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_qhull SOURCE_DIR)
set(QHULL_SOURCE_DIR ${SOURCE_DIR})
