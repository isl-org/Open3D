include(ExternalProject)

ExternalProject_Add(
    ext_openblas
    PREFIX openblas
    GIT_REPOSITORY https://github.com/xianyi/OpenBLAS.git
    GIT_TAG v0.3.10
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    # OpenBLAS builds in source directory.
    BUILD_IN_SOURCE True
    # Avoids libopenblas.so, only use libopenblas.a.
    BUILD_COMMAND make TARGET=NEHALEM NO_SHARED=1
)

ExternalProject_Get_property(ext_openblas SOURCE_DIR)
set(OPENBLAS_INCLUDE_DIR "${SOURCE_DIR}/") # The "/"" is critical, see import_3rdparty_library.
set(OPENBLAS_LIB_DIR "${SOURCE_DIR}")
set(OPENBLAS_LIBRARIES openblas)  # Extended to libopenblas.a automatically.
message(STATUS "OPENBLAS_INCLUDE_DIR: ${OPENBLAS_INCLUDE_DIR}")
message(STATUS "OPENBLAS_LIB_DIR ${OPENBLAS_LIB_DIR}")
message(STATUS "OPENBLAS_LIBRARIES: ${OPENBLAS_LIBRARIES}")
