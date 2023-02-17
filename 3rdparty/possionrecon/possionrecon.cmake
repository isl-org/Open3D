include(ExternalProject)

ExternalProject_Add(
    ext_poisson
    PREFIX poisson
    URL https://github.com/isl-org/Open3D-PoissonRecon/archive/fd273ea8c77a36973d6565a495c9969ccfb12d3b.tar.gz
    URL_HASH SHA256=917d98e037982d57a159fa166b259ff3dc90ffffe09c6a562a71b400f6869ddf
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/poisson"
    SOURCE_DIR "poisson/src/ext_poisson/PoissonRecon" # Add extra directory level for POISSON_INCLUDE_DIRS.
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_poisson SOURCE_DIR)
set(POISSON_INCLUDE_DIRS ${SOURCE_DIR}) # Not using "/" is critical.
