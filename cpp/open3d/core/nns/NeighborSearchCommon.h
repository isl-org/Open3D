// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
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

}  // namespace nns
}  // namespace core
}  // namespace open3d
