include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    # Commit point: https://gitlab.com/libeigen/eigen/-/merge_requests/716
    URL https://gitlab.com/libeigen/eigen/-/archive/da7909592376c893dabbc4b6453a8ffe46b1eb8e/eigen-da7909592376c893dabbc4b6453a8ffe46b1eb8e.tar.gz
    URL_HASH SHA256=37f71e1d7c408e2cc29ef90dcda265e2de0ad5ed6249d7552f6c33897eafd674
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR}/Eigen)
