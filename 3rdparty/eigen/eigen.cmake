include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    # Feb 02, 2022
    URL https://gitlab.com/libeigen/eigen/-/archive/18b50458b6199c93fcc11252e03d91130af077ce/eigen-18b50458b6199c93fcc11252e03d91130af077ce.tar.gz
    URL_HASH SHA256=17E29F5C874D79CE2872A48B16098C7ED22120FF10F1D626477AE59CBD9FF6C3
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/eigen"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_eigen SOURCE_DIR)
set(EIGEN_INCLUDE_DIRS ${SOURCE_DIR}/Eigen)
