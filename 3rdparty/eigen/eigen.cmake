include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.tar.gz
    URL_HASH SHA256=8b9be3abc1bfd89cb8917aa10ea4b68ca6a4bfadcb0be8e8ef9c79114b92871b
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR}/Eigen)
