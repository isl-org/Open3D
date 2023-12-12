include(ExternalProject)

ExternalProject_Add(
    ext_poisson
    PREFIX poisson
    URL https://github.com/isl-org/Open3D-PoissonRecon/archive/90f3f064e275b275cff445881ecee5a7c495c9e0.tar.gz
    URL_HASH SHA256=1310df0c80ff0616b8fcf9b2fb568aa9b2190d0e071b0ead47dba339c146b1d3
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/poisson"
    SOURCE_DIR "poisson/src/ext_poisson/PoissonRecon" # Add extra directory level for POISSON_INCLUDE_DIRS.
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_poisson SOURCE_DIR)
set(POISSON_INCLUDE_DIRS ${SOURCE_DIR}) # Not using "/" is critical.
