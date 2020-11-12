// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <vector>

#include "open3d/core/Atomic.h"
#include "open3d/utility/MiniVec.h"
#include "open3d/utility/ParallelScan.h"

namespace open3d {
namespace ml {
namespace impl {

/// This function voxelizes a point cloud.
/// The function returns the integer coordinates of the voxels that contain
/// points and a compact list of the indices that associate the voxels to the
/// points.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam NDIM    The number of dimensions of the points.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
///
/// \param num_points    The number of points.
///
/// \param points    Array with the point positions. The shape is
///        [num_points,NDIM].
///
/// \param voxel_size    The edge lenghts of the voxel. The shape is [NDIM]
///
/// \param points_range_min    The lower bound of the domain to be voxelized.
///        The shape is [NDIM].
///
/// \param points_range_max    The upper bound of the domain to be voxelized.
///        The shape is [NDIM].
///
/// \param max_points_per_voxel    This parameter limits the number of points
///        that are recorderd for each voxel.
///
/// \param max_voxels    This parameter limits the number of voxels that will
///        be generated.
///
/// \param output_allocator    An object that implements functions for
///         allocating the output arrays. The object must implement functions
///         AllocVoxelCoords(int32_t** ptr, int64_t rows, int64_t cols),
///         AllocVoxelPointIndices(int64_t** ptr, int64_t size), and
///         AllocVoxelPointRowSplits(int64_t** ptr, int64_t size). All
///         functions should allocate memory and return a pointer to that memory
///         in ptr. The argments size, rows, and cols define the size of the
///         array as the number of elements. All functions must accept zero
///         size arguments. In this case ptr does not need to be set.
///
template <class T, int NDIM, class OUTPUT_ALLOCATOR>
void VoxelizeCPU(size_t num_points,
                 const T* const points,
                 const T* const voxel_size,
                 const T* const points_range_min,
                 const T* const points_range_max,
                 const int64_t max_points_per_voxel,
                 const int64_t max_voxels,
                 OUTPUT_ALLOCATOR& output_allocator) {
    using namespace open3d::utility;
    typedef MiniVec<T, NDIM> Vec_t;
    const Vec_t inv_voxel_size = T(1) / Vec_t(voxel_size);
    const Vec_t points_range_min_vec(points_range_min);
    const Vec_t points_range_max_vec(points_range_max);
    MiniVec<int32_t, NDIM> extents =
            ceil((points_range_max_vec - points_range_min_vec) * inv_voxel_size)
                    .template cast<int32_t>();
    MiniVec<int64_t, NDIM> strides;
    for (int i = 0; i < NDIM; ++i) {
        strides[i] = 1;
        for (int j = 0; j < i; ++j) {
            strides[i] *= extents[j];
        }
    }
    const int64_t invalid_hash = strides[NDIM - 1] * extents[NDIM - 1];

    auto CoordFn = [&](const Vec_t& point) {
        auto coords = ((point - points_range_min_vec) * inv_voxel_size)
                              .template cast<int64_t>();
        return coords;
    };

    auto HashFn = [&](const Vec_t& point) {
        if ((point >= points_range_min_vec && point <= points_range_max_vec)
                    .all()) {
            auto coords = CoordFn(point);
            int64_t linear_idx = coords.dot(strides);
            return linear_idx;
        }
        return invalid_hash;
    };

    std::vector<std::pair<int64_t, int64_t>> hashes_indices(num_points);
    tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_points),
                      [&](const tbb::blocked_range<int64_t>& r) {
                          for (int64_t i = r.begin(); i != r.end(); ++i) {
                              Vec_t pos(points + NDIM * i);
                              hashes_indices[i].first = HashFn(pos);
                              hashes_indices[i].second = i;
                          }
                      });
    tbb::parallel_sort(hashes_indices);

    uint64_t num_voxels = 1;
    tbb::parallel_for(tbb::blocked_range<int64_t>(1, hashes_indices.size()),
                      [&](const tbb::blocked_range<int64_t>& r) {
                          for (int64_t i = r.begin(); i != r.end(); ++i) {
                              if (hashes_indices[i - 1].first !=
                                  hashes_indices[i].first) {
                                  core::AtomicFetchAddRelaxed(&num_voxels, 1);
                              }
                          }
                      });
    if (invalid_hash == hashes_indices.back().first) {
        --num_voxels;
    }
    num_voxels = std::min(int64_t(num_voxels), max_voxels);

    int32_t* out_voxel_coords = nullptr;
    output_allocator.AllocVoxelCoords(&out_voxel_coords, num_voxels, NDIM);
    int64_t* out_voxel_row_splits = nullptr;
    output_allocator.AllocVoxelPointRowSplits(&out_voxel_row_splits,
                                              num_voxels + 1);

    std::vector<int64_t> tmp_point_indices;
    {
        int64_t hash_i = 0;  // index into the vector hashes_indices
        for (int64_t voxel_i = 0; voxel_i < num_voxels; ++voxel_i) {
            // compute voxel coord and the prefix sum value
            auto coord = CoordFn(
                    Vec_t(points + hashes_indices[hash_i].second * NDIM));
            for (int d = 0; d < NDIM; ++d) {
                out_voxel_coords[voxel_i * NDIM + d] = coord[d];
            }
            out_voxel_row_splits[voxel_i] = tmp_point_indices.size();

            // add up to max_points_per_voxel indices for this voxel
            int64_t points_per_voxel = 0;
            const int64_t current_hash = hashes_indices[hash_i].first;
            for (; hash_i < hashes_indices.size(); ++hash_i) {
                if (current_hash != hashes_indices[hash_i].first) {
                    // new voxel starts -> break
                    break;
                }
                if (points_per_voxel < max_points_per_voxel) {
                    tmp_point_indices.push_back(hashes_indices[hash_i].second);
                    ++points_per_voxel;
                }
            }
        }
        out_voxel_row_splits[num_voxels] = tmp_point_indices.size();
    }
    int64_t* out_point_indices = nullptr;
    output_allocator.AllocVoxelPointIndices(&out_point_indices,
                                            tmp_point_indices.size());
    memcpy(out_point_indices, tmp_point_indices.data(),
           tmp_point_indices.size() * sizeof(int64_t));
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
