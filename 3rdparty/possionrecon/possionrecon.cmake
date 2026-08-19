include(ExternalProject)

ExternalProject_Add(
    ext_poisson
    PREFIX poisson
    URL https://github.com/isl-org/Open3D-PoissonRecon/archive/24c9c88c1404edc43c29b9e770fc4cb8155c15b2.tar.gz
    URL_HASH SHA256=a8409aa218bedc239d0768299fa7e63287e00b1cc632457a051bc3a91e372cab
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/poisson"
    SOURCE_DIR "poisson/src/ext_poisson/PoissonRecon" # Add extra directory level for POISSON_INCLUDE_DIRS.
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_poisson SOURCE_DIR)
set(POISSON_INCLUDE_DIRS ${SOURCE_DIR}) # Not using "/" is critical.
