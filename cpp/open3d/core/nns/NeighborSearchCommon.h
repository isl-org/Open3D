// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace core {
namespace nns {

/// Supported metrics
enum Metric { L1, L2, Linf };

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/// Spatial hashing function for integer coordinates.
HOST_DEVICE inline size_t SpatialHash(int x, int y, int z) {
    return x * 73856096 ^ y * 193649663 ^ z * 83492791;
}

HOST_DEVICE inline size_t SpatialHash(const utility::MiniVec<int, 3> &xyz) {
    return SpatialHash(xyz[0], xyz[1], xyz[2]);
}

/// Computes an integer voxel index for a 3D position.
///
/// \param pos               A 3D position.
/// \param inv_voxel_size    The reciprocal of the voxel size
///
template <class TVecf>
HOST_DEVICE inline utility::MiniVec<int, 3> ComputeVoxelIndex(
        const TVecf &pos, const typename TVecf::Scalar_t &inv_voxel_size) {
    TVecf ref_coord = pos * inv_voxel_size;

    utility::MiniVec<int, 3> voxel_index;
    voxel_index = floor(ref_coord).template cast<int>();
    return voxel_index;
}
#undef HOST_DEVICE

/// Base struct for NanoFlann index holder
struct NanoFlannIndexHolderBase {
    virtual ~NanoFlannIndexHolderBase() {}
};

// ── SYCL NNS tunable defaults ─────────────────────────────────────────────
// These constants serve as constructor defaults for KnnIndex and
// FixedRadiusIndex when the search runs on a SYCL device.  They are defined
// here so that the public index headers can reference them without pulling in
// any SYCL headers.
//
// Hardware reference – Intel Panther Lake Arc iGPU (2025):
//   92 KB SLM per EU, 12 EUs per tile, ~2.5 TFLOPS FP32.
//   A 4 MiB distance tile fits in the combined LLC of 12 EUs (~46 MiB),
//   keeping memory traffic on-die and maximising AddMM throughput.
//
// Tuning guidance:
//   kSYCLKnnDefaultTileBytes
//     Integrated GPU (Iris Xe, Arc iGPU, Panther Lake): 4–8 MiB keeps the
//       distance tile resident in the last-level cache, reducing DRAM traffic.
//     Discrete GPU (Arc Alchemist / Battlemage, Data-Centre GPUs): 16–32 MiB
//       improves oneMKL GEMM efficiency; these parts have ~800 GB/s DRAM
//       bandwidth and benefit from larger tiles.
//   kSYCLKnnSmallKMax
//     Maximum k for the register-resident heap path.  For Intel Xe in 128-GRF
//     mode, ~32 float/int register pairs fit before spilling to scratch memory.
//     Raise to 64 on parts with 256-GRF support; lower if register pressure
//     causes kernel occupancy to drop.
//   kSYCLKnnMidKMax
//     Maximum k for the template-dispatched scratch-resident heap path.
//     k > kSYCLKnnMidKMax falls back to a sequential oneDPL partial_sort.
//     Increase to 1024 if very large k is common and scratch bandwidth allows.

/// Default distance-tile budget in bytes (8 MiB, tuned for iGPU).
constexpr int64_t kSYCLKnnDefaultTileBytes = 8LL * 1024 * 1024;

/// Upper bound of k for the GRF-register heap path (eliminates scratch spill).
constexpr int64_t kSYCLKnnSmallKMax = 32;

/// Upper bound of k for the proportional scratch-resident heap path.
constexpr int64_t kSYCLKnnMidKMax = 512;

}  // namespace nns
}  // namespace core
}  // namespace open3d
