// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <nanoflann.hpp>

#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace ml {
namespace impl {

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

HOST_DEVICE inline size_t SpatialHash(const utility::MiniVec<int, 3>& xyz) {
    return SpatialHash(xyz[0], xyz[1], xyz[2]);
}

/// Computes an integer voxel index for a 3D position.
///
/// \param pos               A 3D position.
/// \param inv_voxel_size    The reciprocal of the voxel size
///
template <class TVecf>
HOST_DEVICE inline utility::MiniVec<int, 3> ComputeVoxelIndex(
        const TVecf& pos, const typename TVecf::Scalar_t& inv_voxel_size) {
    TVecf ref_coord = pos * inv_voxel_size;

    utility::MiniVec<int, 3> voxel_index;
    voxel_index = floor(ref_coord).template cast<int>();
    return voxel_index;
}
#undef HOST_DEVICE

/// Adaptor for nanoflann
template <class T>
class Adaptor {
public:
    Adaptor(size_t num_points, const T* const data)
        : num_points(num_points), data(data) {}

    inline size_t kdtree_get_point_count() const { return num_points; }

    inline T kdtree_get_pt(const size_t idx, int dim) const {
        return data[3 * idx + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }

private:
    size_t num_points;
    const T* const data;
};

template <int METRIC, class T>
struct SelectNanoflannAdaptor {};

template <class T>
struct SelectNanoflannAdaptor<L2, T> {
    typedef nanoflann::L2_Adaptor<T, Adaptor<T>> Adaptor_t;
};

template <class T>
struct SelectNanoflannAdaptor<L1, T> {
    typedef nanoflann::L1_Adaptor<T, Adaptor<T>> Adaptor_t;
};

}  // namespace impl
}  // namespace ml
}  // namespace open3d
