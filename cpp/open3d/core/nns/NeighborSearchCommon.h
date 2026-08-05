// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file NeighborSearchCommon.h
/// \brief Shared types and SYCL nearest-neighbor search tuning defaults.

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

/// SYCL NNS defaults for \ref KnnIndex and \ref FixedRadiusIndex constructors.

/// Default distance-tile budget in bytes (8 MiB, tuned for iGPU last-level
/// cache). Discrete GPUs cache is larger and we can increase to 16-32 MiB.
constexpr int64_t kSYCLKnnDefaultTileBytes = 8LL * 1024 * 1024;

/// Upper bound of k for the GRF-register heap path (eliminates scratch spill).
// ~32 on 128-GRF Xe, raise to 64 on 256-GRF Xe. Reduce if kernel occupancy
// drops.
constexpr int64_t kSYCLKnnSmallKMax = 32;

/// Upper bound of k for the proportional scratch-resident heap path. Larger k
/// uses sequential oneDPL partial_sort.
constexpr int64_t kSYCLKnnMidKMax = 512;

}  // namespace nns
}  // namespace core
}  // namespace open3d
