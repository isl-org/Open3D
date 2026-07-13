include(ExternalProject)

# SYCL*TLA (SYCL Templates for Linear Algebra) is a fork of NVIDIA CUTLASS
# that extends the CUTLASS/CuTe API to Intel GPUs via SYCL.
# Version v0.9.1 is based on CUTLASS v4.2.1, matching Open3D's CUTLASS version.
# Used by the SYCL ML ops path (BUILD_SYCL_MODULE=ON) for future Intel GPU support.
# Actual CUDA→SYCL kernel porting is handled separately.
ExternalProject_Add(
    ext_sycl_tla
    PREFIX sycl_tla
    URL https://github.com/intel/sycl-tla/archive/refs/tags/v0.9.1.tar.gz
    URL_HASH SHA256=407d85b4358294ddae19e03d6b99af611660e7c0a24f8cbe5c24546dfe7ac86e
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/sycl_tla"
    UPDATE_COMMAND ""
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
