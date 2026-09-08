include(ExternalProject)

# SYCL*TLA (SYCL Templates for Linear Algebra) is a fork of NVIDIA CUTLASS
# that extends the CUTLASS/CuTe API to Intel GPUs via SYCL.
# Pinned to a sycl-tla main commit while validating the device-agnostic
# float32 GEMM path. Used by the SYCL ML ops path (BUILD_SYCL_MODULE=ON) for
# future Intel GPU support. Actual CUDA→SYCL kernel porting is handled
# separately.
# Downloaded as a GitHub commit zip archive (rather than a git clone) so the
# source is fetched in one HTTP request with a verifiable SHA256, matching
# the pattern used by the other 3rdparty deps in this file (e.g. assimp).
# No submodules exist at this commit, so a plain zip is a complete checkout.
ExternalProject_Add(
    ext_sycl_tla
    PREFIX sycl_tla
    URL https://github.com/intel/sycl-tla/archive/122b0676698dbb437db09705e5ae2e2c57376f8a.zip
    URL_HASH SHA256=8feba59b6934b0be61cd721b936cd82fbf3ba263dcd0aae5c92b3510ad960285
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/sycl_tla"
    PATCH_COMMAND
        ${CMAKE_COMMAND}
        -DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/0001-fix-oneapi-2025.3-ieee-gemm.patch
        -DSOURCE_DIR=<SOURCE_DIR>
        -P ${CMAKE_CURRENT_LIST_DIR}/../librealsense/apply_patch.cmake
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
