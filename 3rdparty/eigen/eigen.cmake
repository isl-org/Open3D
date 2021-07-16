include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.tar.bz2
    URL_HASH SHA256=92641cb17a92bcf311c7fc095a555ca0e32990503573dda80eb1764dc37dcac9
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR}/Eigen)
