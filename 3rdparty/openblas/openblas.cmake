include(ExternalProject)

set(OPENBLAS_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/openblas)

if(LINUX_AARCH64 OR APPLE_AARCH64)
    set(OPENBLAS_TARGET "ARMV8")
else()
    set(OPENBLAS_TARGET "NEHALEM")
endif()

set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_INSTALL_PREFIX}/include/openblas/") # The "/"" is critical, see open3d_import_3rdparty_library.
set(OPENBLAS_LIB_DIR "${OPENBLAS_INSTALL_PREFIX}/lib")
set(OPENBLAS_LIBRARIES openblas)  # Extends to libopenblas.a automatically.

ExternalProject_Add(
    ext_openblas
    PREFIX openblas
    URL https://github.com/xianyi/OpenBLAS/releases/download/v0.3.19/OpenBLAS-0.3.19.tar.gz
    URL_HASH SHA256=947f51bfe50c2a0749304fbe373e00e7637600b0a47b78a51382aeb30ca08562
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/openblas"
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS}
        -DTARGET=${OPENBLAS_TARGET}
        -DCMAKE_INSTALL_PREFIX=${OPENBLAS_INSTALL_PREFIX}
    BUILD_BYPRODUCTS ${OPENBLAS_LIB_DIR}/libopenblas.a
)

message(STATUS "OPENBLAS_INCLUDE_DIR: ${OPENBLAS_INCLUDE_DIR}")
message(STATUS "OPENBLAS_LIB_DIR ${OPENBLAS_LIB_DIR}")
message(STATUS "OPENBLAS_LIBRARIES: ${OPENBLAS_LIBRARIES}")
