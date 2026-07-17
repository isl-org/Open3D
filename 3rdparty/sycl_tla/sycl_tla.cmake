include(ExternalProject)

# SYCL*TLA (SYCL Templates for Linear Algebra) is a fork of NVIDIA CUTLASS
# that extends the CUTLASS/CuTe API to Intel GPUs via SYCL.
# Track sycl-tla main while validating the device-agnostic float32 GEMM path.
# Used by the SYCL ML ops path (BUILD_SYCL_MODULE=ON) for future Intel GPU
# support. Actual CUDA→SYCL kernel porting is handled separately.
ExternalProject_Add(
    ext_sycl_tla
    PREFIX sycl_tla
    GIT_REPOSITORY https://github.com/intel/sycl-tla.git
    GIT_TAG main
    GIT_SHALLOW TRUE
    UPDATE_DISCONNECTED TRUE
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/sycl_tla"
    PATCH_COMMAND
        /bin/bash ${CMAKE_CURRENT_LIST_DIR}/apply_patch.sh
        ${CMAKE_CURRENT_LIST_DIR}/0001-fix-oneapi-2025.3-ieee-gemm.patch
        <SOURCE_DIR>
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_sycl_tla SOURCE_DIR)
# sycl-tla headers live in include/, mirroring CUTLASS v4 layout. The
# device::GemmUniversalAdapter path additionally pulls in helper headers
# (e.g. cutlass/util/packed_stride.hpp, cutlass/util/sycl_event_manager.hpp)
# that live under tools/util/include/ (CUTLASS's separate "tools" utility
# headers directory, not bundled into include/), so both must be on the
# include path.
set(SYCL_TLA_INCLUDE_DIRS ${SOURCE_DIR}/include/
                          ${SOURCE_DIR}/tools/util/include/)
