include(ExternalProject)

ExternalProject_Add(
    ext_poisson
    PREFIX poisson
    URL https://github.com/isl-org/Open3D-PoissonRecon/archive/6ddec9a69f4aeb7a715e8f496310929d9f493041.tar.gz
    URL_HASH SHA256=3e02bebd1b22f76bb8874be6ff7ab60a0f74ed690829befe0f90e0f2b70bbbe6
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/poisson"
    SOURCE_DIR "poisson/src/ext_poisson/PoissonRecon" # Add extra directory level for POISSON_INCLUDE_DIRS.
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_poisson SOURCE_DIR)
set(POISSON_INCLUDE_DIRS ${SOURCE_DIR}) # Not using "/" is critical.
