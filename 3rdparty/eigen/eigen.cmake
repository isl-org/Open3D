include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3/eigen-3.3.tar.gz
    URL_HASH SHA256=1bc843b08d0015d5cc0d403662ee182366cb19b0859fab8872585342c671ec12
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR}/Eigen)
