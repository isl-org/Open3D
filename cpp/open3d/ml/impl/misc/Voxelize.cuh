// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include <thrust/sequence.h>

#include <cub/cub.cuh>

#include "open3d/ml/impl/misc/MemoryAllocation.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/MiniVec.h"

namespace open3d {
namespace ml {
namespace impl {

namespace {

using namespace open3d::utility;

template <class T, bool LARGE_ARRAY>
__global__ void IotaCUDAKernel(T* first, int64_t len, T value) {
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    if (LARGE_ARRAY) {
        const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
        const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
        linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                     y * gridDim.x * blockDim.x + x;
    } else {
        linear_idx = x;
    }
    if (linear_idx < len) {
        T* ptr = first + linear_idx;
        value += linear_idx;
        *ptr = value;
    }
}

/// Iota function for CUDA
template <class T>
void IotaCUDA(const cudaStream_t& stream, T* first, T* last, T value) {
    ptrdiff_t len = last - first;
    if (len) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        if (len > block.x * INT32_MAX) {
            grid.y = std::ceil(std::cbrt(len));
            grid.z = grid.y;
            grid.x = DivUp(len, int64_t(grid.z) * grid.y * block.x);
            IotaCUDAKernel<T, true>
                    <<<grid, block, 0, stream>>>(first, len, value);
        } else {
            grid = dim3(DivUp(len, block.x), 1, 1);
            IotaCUDAKernel<T, false>
                    <<<grid, block, 0, stream>>>(first, len, value);
        }
    }
}

__global__ void ComputeBatchIdKernel(int64_t* hashes,
                                     const int64_t num_voxels,
                                     const int64_t batch_hash) {
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= num_voxels) return;

    hashes[linear_idx] /= batch_hash;
}

/// This function computes batch_id from hash value.
///
/// \param hashes    Input and output array.
/// \param num_voxels    Number of valid voxels.
/// \param batch_hash    The value used to hash batch dimension.
///
void ComputeBatchId(const cudaStream_t& stream,
                    int64_t* hashes,
                    const int64_t num_voxels,
                    const int64_t batch_hash) {
    if (num_voxels) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num_voxels));
        grid.z = grid.y;
        grid.x = DivUp(num_voxels, int64_t(grid.z) * grid.y * block.x);
        ComputeBatchIdKernel<<<grid, block, 0, stream>>>(hashes, num_voxels,
                                                         batch_hash);
    }
}

__global__ void ComputeVoxelPerBatchKernel(int64_t* num_voxels_per_batch,
                                           int64_t* unique_batches_count,
                                           int64_t* unique_batches,
                                           const int64_t num_batches) {
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= num_batches) return;

    int64_t out_idx = unique_batches[linear_idx];
    num_voxels_per_batch[out_idx] = unique_batches_count[linear_idx];
}

/// This function computes number of voxels per batch element.
///
/// \param num_voxels_per_batch    The output array.
/// \param unique_batches_count    Counts for unique batch_id.
/// \param unique_batches    Unique batch_id.
/// \param num_batches    Number of non empty batches (<= batch_size).
///
void ComputeVoxelPerBatch(const cudaStream_t& stream,
                          int64_t* num_voxels_per_batch,
                          int64_t* unique_batches_count,
                          int64_t* unique_batches,
                          const int64_t num_batches) {
    if (num_batches) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num_batches));
        grid.z = grid.y;
        grid.x = DivUp(num_batches, int64_t(grid.z) * grid.y * block.x);
        ComputeVoxelPerBatchKernel<<<grid, block, 0, stream>>>(
                num_voxels_per_batch, unique_batches_count, unique_batches,
                num_batches);
    }
}

__global__ void ComputeIndicesBatchesKernel(int64_t* indices_batches,
                                            const int64_t* row_splits,
                                            const int64_t batch_size) {
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= batch_size) return;

    for (int64_t i = row_splits[linear_idx]; i < row_splits[linear_idx + 1];
         ++i) {
        indices_batches[i] = linear_idx;
    }
}

/// This function computes mapping of index to batch_id.
///
/// \param indices_batches    The output array.
/// \param row_splits    The row_splits for defining batches.
/// \param batch_size    The batch_size of given points.
///
void ComputeIndicesBatches(const cudaStream_t& stream,
                           int64_t* indices_batches,
                           const int64_t* row_splits,
                           const int64_t batch_size) {
    if (batch_size) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(batch_size));
        grid.z = grid.y;
        grid.x = DivUp(batch_size, int64_t(grid.z) * grid.y * block.x);
        ComputeIndicesBatchesKernel<<<grid, block, 0, stream>>>(
                indices_batches, row_splits, batch_size);
    }
}

template <class T, int NDIM>
__global__ void ComputeHashKernel(
        int64_t* __restrict__ hashes,
        int64_t num_points,
        const T* const __restrict__ points,
        const int64_t batch_size,
        const int64_t* row_splits,
        const int64_t* indices_batches,
        const open3d::utility::MiniVec<T, NDIM> points_range_min_vec,
        const open3d::utility::MiniVec<T, NDIM> points_range_max_vec,
        const open3d::utility::MiniVec<T, NDIM> inv_voxel_size,
        const open3d::utility::MiniVec<int64_t, NDIM> strides,
        const int64_t batch_hash,
        const int64_t invalid_hash) {
    typedef MiniVec<T, NDIM> Vec_t;
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= num_points) return;

    Vec_t point(points + linear_idx * NDIM);

    if ((point >= points_range_min_vec && point <= points_range_max_vec)
                .all()) {
        auto coords = ((point - points_range_min_vec) * inv_voxel_size)
                              .template cast<int64_t>();
        int64_t h = coords.dot(strides);
        h += indices_batches[linear_idx] * batch_hash;  // add hash for batch_id
        hashes[linear_idx] = h;
    } else {
        hashes[linear_idx] = invalid_hash;  // max hash value used as invalid
    }
}

/// This function computes the hash (linear index) for each point.
/// Points outside the range will get a specific hash value.
///
/// \tparam T    The floating point type for the points
/// \tparam NDIM    The number of dimensions, e.g., 3.
///
/// \param hashes    The output vector with the hashes/linear indexes.
/// \param num_points    The number of points.
/// \param points    The array with the point coordinates. The shape is
///        [num_points,NDIM] and the storage order is row-major.
/// \param batch_size    The batch size of points.
/// \param row_splits    row_splits for defining batches.
/// \param indices_batches    Mapping of index to batch_id.
/// \param points_range_min_vec    The minimum range for a point to be valid.
/// \param points_range_max_vec    The maximum range for a point to be valid.
/// \param inv_voxel_size    The reciprocal of the voxel edge lengths in each
///        dimension
/// \param strides    The strides for computing the linear index.
/// \param batch_hash    The value for hashing batch dimension.
/// \param invalid_hash    The value to use for points outside the range.
template <class T, int NDIM>
void ComputeHash(const cudaStream_t& stream,
                 int64_t* hashes,
                 int64_t num_points,
                 const T* const points,
                 const int64_t batch_size,
                 const int64_t* row_splits,
                 const int64_t* indices_batches,
                 const MiniVec<T, NDIM> points_range_min_vec,
                 const MiniVec<T, NDIM> points_range_max_vec,
                 const MiniVec<T, NDIM> inv_voxel_size,
                 const MiniVec<int64_t, NDIM> strides,
                 const int64_t batch_hash,
                 const int64_t invalid_hash) {
    if (num_points) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num_points));
        grid.z = grid.y;
        grid.x = DivUp(num_points, int64_t(grid.z) * grid.y * block.x);
        ComputeHashKernel<T, NDIM><<<grid, block, 0, stream>>>(
                hashes, num_points, points, batch_size, row_splits,
                indices_batches, points_range_min_vec, points_range_max_vec,
                inv_voxel_size, strides, batch_hash, invalid_hash);
    }
}

template <class T>
__global__ void LimitCountsKernel(T* counts, int64_t num, T limit) {
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= num) return;

    if (counts[linear_idx] > limit) {
        counts[linear_idx] = limit;
    }
}

/// This function performs an element-wise minimum operation.
///
/// \param counts    The input and output array.
/// \param num    Number of input elements.
/// \param limit    The second operator for the minimum operation.
template <class T>
void LimitCounts(const cudaStream_t& stream, T* counts, int64_t num, T limit) {
    if (num) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num));
        grid.z = grid.y;
        grid.x = DivUp(num, int64_t(grid.z) * grid.y * block.x);
        LimitCountsKernel<<<grid, block, 0, stream>>>(counts, num, limit);
    }
}

__global__ void ComputeStartIdxKernel(
        int64_t* start_idx,
        int64_t* points_count,
        const int64_t* num_voxels_prefix_sum,
        const int64_t* unique_hashes_count_prefix_sum,
        const int64_t* out_batch_splits,
        const int64_t batch_size,
        const int64_t max_voxels,
        const int64_t max_points_per_voxel) {
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= batch_size) return;

    int64_t voxel_idx;
    if (0 == linear_idx) {
        voxel_idx = 0;
    } else {
        voxel_idx = num_voxels_prefix_sum[linear_idx - 1];
    }

    int64_t begin_out = out_batch_splits[linear_idx];
    int64_t end_out = out_batch_splits[linear_idx + 1];

    for (int64_t out_idx = begin_out; out_idx < end_out;
         out_idx++, voxel_idx++) {
        if (voxel_idx == 0) {
            start_idx[out_idx] = 0;
            points_count[out_idx] = min(max_points_per_voxel,
                                        unique_hashes_count_prefix_sum[0]);
        } else {
            start_idx[out_idx] = unique_hashes_count_prefix_sum[voxel_idx - 1];
            points_count[out_idx] =
                    min(max_points_per_voxel,
                        unique_hashes_count_prefix_sum[voxel_idx] -
                                unique_hashes_count_prefix_sum[voxel_idx - 1]);
        }
    }
}

/// Computes the starting index of each voxel.
///
/// \param start_idx The output array for storing starting index.
/// \param points_count The output array for storing points count.
/// \param num_voxels_prefix_sum The Inclusive prefix sum which gives
///        the index of starting voxel for each batch.
/// \param unique_hashes_count_prefix_sum Inclusive prefix sum defining
///        where point indices for each voxel ends.
/// \param out_batch_splits Defines starting and ending voxels for
///        each batch element.
/// \param batch_size The batch size.
/// \param max_voxels Maximum voxels per batch.
/// \param max_points_per_voxel Maximum points per voxel.
///
void ComputeStartIdx(const cudaStream_t& stream,
                     int64_t* start_idx,
                     int64_t* points_count,
                     const int64_t* num_voxels_prefix_sum,
                     const int64_t* unique_hashes_count_prefix_sum,
                     const int64_t* out_batch_splits,
                     const int64_t batch_size,
                     const int64_t max_voxels,
                     const int64_t max_points_per_voxel) {
    if (batch_size) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(batch_size));
        grid.z = grid.y;
        grid.x = DivUp(batch_size, int64_t(grid.z) * grid.y * block.x);
        ComputeStartIdxKernel<<<grid, block, 0, stream>>>(
                start_idx, points_count, num_voxels_prefix_sum,
                unique_hashes_count_prefix_sum, out_batch_splits, batch_size,
                max_voxels, max_points_per_voxel);
    }
}

template <class T, int NDIM>
__global__ void ComputeVoxelCoordsKernel(
        int32_t* __restrict__ voxel_coords,
        const T* const __restrict__ points,
        const int64_t* const __restrict__ point_indices,
        const int64_t* const __restrict__ prefix_sum,
        const MiniVec<T, NDIM> points_range_min_vec,
        const MiniVec<T, NDIM> inv_voxel_size,
        int64_t num_voxels) {
    typedef MiniVec<T, NDIM> Vec_t;
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= num_voxels) return;

    int64_t point_idx = point_indices[prefix_sum[linear_idx]];

    Vec_t point(points + point_idx * NDIM);
    auto coords = ((point - points_range_min_vec) * inv_voxel_size)
                          .template cast<int32_t>();

    for (int i = 0; i < NDIM; ++i) {
        voxel_coords[linear_idx * NDIM + i] = coords[i];
    }
}

/// Computes the coordinates for each voxel
///
/// \param voxel_coords    The output array with shape [num_voxels, NDIM].
/// \param points    The array with the point coordinates.
/// \param point_indices    The array with the point indices for all voxels.
/// \param prefix_sum    Inclusive prefix sum defining where the point indices
///        for each voxels end.
/// \param points_range_min    The lower bound of the domain to be
///        voxelized.
/// \param points_range_max    The upper bound of the domain to be
///        voxelized.
/// \param inv_voxel_size    The reciprocal of the voxel edge lengths for each
///        dimension.
/// \param num_voxels    The number of voxels.
template <class T, int NDIM>
void ComputeVoxelCoords(const cudaStream_t& stream,
                        int32_t* voxel_coords,
                        const T* const points,
                        const int64_t* const point_indices,
                        const int64_t* const prefix_sum,
                        const MiniVec<T, NDIM> points_range_min_vec,
                        const MiniVec<T, NDIM> inv_voxel_size,
                        int64_t num_voxels) {
    if (num_voxels) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num_voxels));
        grid.z = grid.y;
        grid.x = DivUp(num_voxels, int64_t(grid.z) * grid.y * block.x);
        ComputeVoxelCoordsKernel<<<grid, block, 0, stream>>>(
                voxel_coords, points, point_indices, prefix_sum,
                points_range_min_vec, inv_voxel_size, num_voxels);
    }
}

__global__ void CopyPointIndicesKernel(
        int64_t* __restrict__ out,
        const int64_t* const __restrict__ point_indices,
        const int64_t* const __restrict__ prefix_sum_in,
        const int64_t* const __restrict__ prefix_sum_out,
        const int64_t num_voxels) {
    // TODO data coalescing can be optimized
    int64_t linear_idx;
    const int64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const int64_t z = blockDim.z * blockIdx.z + threadIdx.z;
    linear_idx = z * gridDim.x * blockDim.x * gridDim.y +
                 y * gridDim.x * blockDim.x + x;

    if (linear_idx >= num_voxels) return;

    int64_t begin_out;
    if (0 == linear_idx) {
        begin_out = 0;
    } else {
        begin_out = prefix_sum_out[linear_idx - 1];
    }
    int64_t end_out = prefix_sum_out[linear_idx];

    int64_t num_points = end_out - begin_out;

    int64_t in_idx = prefix_sum_in[linear_idx];

    for (int64_t out_idx = begin_out; out_idx < end_out; ++out_idx, ++in_idx) {
        out[out_idx] = point_indices[in_idx];
    }
}

/// Copies the point indices for each voxel to the output.
///
/// \param out    The output array with the point indices for all voxels.
/// \param point_indices    The array with the point indices for all voxels.
/// \param prefix_sum_in    Inclusive prefix sum defining where the point
///        indices for each voxels end.
/// \param prefix_sum_out    Inclusive prefix sum defining where the point
///        indices for each voxels end.
/// \param num_voxels    The number of voxels.
///
void CopyPointIndices(const cudaStream_t& stream,
                      int64_t* out,
                      const int64_t* const point_indices,
                      const int64_t* const prefix_sum_in,
                      const int64_t* const prefix_sum_out,
                      const int64_t num_voxels) {
    if (num_voxels) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num_voxels));
        grid.z = grid.y;
        grid.x = DivUp(num_voxels, int64_t(grid.z) * grid.y * block.x);
        CopyPointIndicesKernel<<<grid, block, 0, stream>>>(
                out, point_indices, prefix_sum_in, prefix_sum_out, num_voxels);
    }
}

}  // namespace

/// This function voxelizes a point cloud.
/// The function returns the integer coordinates of the voxels that
/// contain points and a compact list of the indices that associate the
/// voxels to the points.
///
/// All pointer arguments point to device memory unless stated
/// otherwise.
///
/// \tparam T    Floating-point data type for the point positions.
///
/// \tparam NDIM    The number of dimensions of the points.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
/// \param stream    The cuda stream for all kernel launches.
///
/// \param temp    Pointer to temporary memory. If nullptr then the required
///        size of temporary memory will be written to \p temp_size and no
///        work is done.
///
/// \param temp_size    The size of the temporary memory in bytes. This is
///        used as an output if temp is nullptr
///
/// \param texture_alignment    The texture alignment in bytes. This is used
///        for allocating segments within the temporary memory.
///
/// \param num_points    The number of points.
///
/// \param points    Array with the point positions. The shape is
///        [num_points,NDIM].
///
/// \param batch_size    The batch size of points.
///
/// \param row_splits    row_splits for defining batches.
///
/// \param voxel_size    The edge lengths of the voxel. The shape is
///        [NDIM]. This pointer points to host memory!
///
/// \param points_range_min    The lower bound of the domain to be
/// voxelized.
///        The shape is [NDIM].
///        This pointer points to host memory!
///
/// \param points_range_max    The upper bound of the domain to be
/// voxelized.
///        The shape is [NDIM].
///        This pointer points to host memory!
///
/// \param max_points_per_voxel    This parameter limits the number of
/// points
///        that are recorderd for each voxel.
///
/// \param max_voxels    This parameter limits the number of voxels that
///        will be generated.
///
/// \param output_allocator    An object that implements functions for
///         allocating the output arrays. The object must implement
///         functions AllocVoxelCoords(int32_t** ptr, int64_t rows,
///         int64_t cols), AllocVoxelPointIndices(int64_t** ptr, int64_t
///         size), AllocVoxelPointRowSplits(int64_t** ptr, int64_t
///         size) and AllocVoxelBatchSplits(int64_t** ptr, int64_t size).
///         All functions should allocate memory and return a pointer
///         to that memory in ptr. The arguments size, rows, and cols
///         define the size of the array as the number of elements.
///         All functions must accept zero size arguments. In this case
///         ptr does not need to be set.
///
template <class T, int NDIM, class OUTPUT_ALLOCATOR>
void VoxelizeCUDA(const cudaStream_t& stream,
                  void* temp,
                  size_t& temp_size,
                  int texture_alignment,
                  size_t num_points,
                  const T* const points,
                  const size_t batch_size,
                  const int64_t* const row_splits,
                  const T* const voxel_size,
                  const T* const points_range_min,
                  const T* const points_range_max,
                  const int64_t max_points_per_voxel,
                  const int64_t max_voxels,
                  OUTPUT_ALLOCATOR& output_allocator) {
    using namespace open3d::utility;
    typedef MiniVec<T, NDIM> Vec_t;

    const bool get_temp_size = !temp;

    if (get_temp_size) {
        temp = (char*)1;  // worst case pointer alignment
        temp_size = std::numeric_limits<int64_t>::max();
    }

    MemoryAllocation mem_temp(temp, temp_size, texture_alignment);

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
    const int64_t batch_hash = strides[NDIM - 1] * extents[NDIM - 1];
    const int64_t invalid_hash = batch_hash * batch_size;

    /// store batch_id for each point
    std::pair<int64_t*, size_t> indices_batches =
            mem_temp.Alloc<int64_t>(num_points);
    if (!get_temp_size) {
        ComputeIndicesBatches(stream, indices_batches.first, row_splits,
                              batch_size);
    }

    // use double buffers for the sorting
    std::pair<int64_t*, size_t> point_indices =
            mem_temp.Alloc<int64_t>(num_points);
    std::pair<int64_t*, size_t> point_indices_alt =
            mem_temp.Alloc<int64_t>(num_points);
    std::pair<int64_t*, size_t> hashes = mem_temp.Alloc<int64_t>(num_points);
    std::pair<int64_t*, size_t> hashes_alt =
            mem_temp.Alloc<int64_t>(num_points);

    cub::DoubleBuffer<int64_t> point_indices_dbuf(point_indices.first,
                                                  point_indices_alt.first);
    cub::DoubleBuffer<int64_t> hashes_dbuf(hashes.first, hashes_alt.first);

    if (!get_temp_size) {
        IotaCUDA(stream, point_indices.first,
                 point_indices.first + point_indices.second, int64_t(0));
        ComputeHash(stream, hashes.first, num_points, points, batch_size,
                    row_splits, indices_batches.first, points_range_min_vec,
                    points_range_max_vec, inv_voxel_size, strides, batch_hash,
                    invalid_hash);
    }

    {
        // TODO compute end_bit for radix sort
        std::pair<void*, size_t> sort_pairs_temp(nullptr, 0);
        cub::DeviceRadixSort::SortPairs(
                sort_pairs_temp.first, sort_pairs_temp.second, hashes_dbuf,
                point_indices_dbuf, num_points, 0, sizeof(int64_t) * 8, stream);
        sort_pairs_temp = mem_temp.Alloc(sort_pairs_temp.second);
        if (!get_temp_size) {
            cub::DeviceRadixSort::SortPairs(sort_pairs_temp.first,
                                            sort_pairs_temp.second, hashes_dbuf,
                                            point_indices_dbuf, num_points, 0,
                                            sizeof(int64_t) * 8, stream);
        }
        mem_temp.Free(sort_pairs_temp);
    }

    // reuse the alternate buffers
    std::pair<int64_t*, size_t> unique_hashes(hashes_dbuf.Alternate(),
                                              hashes.second);
    std::pair<int64_t*, size_t> unique_hashes_count(
            point_indices_dbuf.Alternate(), point_indices.second);

    // encode unique hashes(voxels) and their counts(points per voxel)
    int64_t num_voxels = 0;
    int64_t last_hash = 0;  // 0 is a valid hash value
    {
        std::pair<void*, size_t> encode_temp(nullptr, 0);
        std::pair<int64_t*, size_t> num_voxels_mem = mem_temp.Alloc<int64_t>(1);

        cub::DeviceRunLengthEncode::Encode(
                encode_temp.first, encode_temp.second, hashes_dbuf.Current(),
                unique_hashes.first, unique_hashes_count.first,
                num_voxels_mem.first, num_points, stream);

        encode_temp = mem_temp.Alloc(encode_temp.second);
        if (!get_temp_size) {
            cub::DeviceRunLengthEncode::Encode(
                    encode_temp.first, encode_temp.second,
                    hashes_dbuf.Current(), unique_hashes.first,
                    unique_hashes_count.first, num_voxels_mem.first, num_points,
                    stream);

            // get the number of voxels
            cudaMemcpyAsync(&num_voxels, num_voxels_mem.first, sizeof(int64_t),
                            cudaMemcpyDeviceToHost, stream);
            // get the last hash value
            cudaMemcpyAsync(&last_hash,
                            hashes_dbuf.Current() + hashes.second - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            // wait for the async copies
            while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
            }
        }
        mem_temp.Free(encode_temp);
    }
    if (invalid_hash == last_hash) {
        // the last hash is invalid we have one voxel less
        --num_voxels;
    }

    // reuse the hashes buffer
    std::pair<int64_t*, size_t> unique_hashes_count_prefix_sum(
            hashes_dbuf.Current(), hashes.second);

    // compute the prefix sum for unique_hashes_count
    // gives starting index of each voxel
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);

        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                unique_hashes_count.first, unique_hashes_count_prefix_sum.first,
                unique_hashes_count.second, stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);
        if (!get_temp_size) {
            // We only need the prefix sum for the first num_voxels.
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    unique_hashes_count.first,
                    unique_hashes_count_prefix_sum.first, num_voxels, stream);
        }
        mem_temp.Free(inclusive_scan_temp);
    }

    // Limit the number of output points to max_points_per_voxel by
    // limiting the unique_hashes_count.
    if (!get_temp_size) {
        if (max_points_per_voxel < num_points) {
            LimitCounts(stream, unique_hashes_count.first, num_voxels,
                        max_points_per_voxel);
        }
    }

    // Convert unique_hashes to batch_id (divide with batch_hash)
    int64_t* unique_hashes_batch_id = unique_hashes.first;
    if (!get_temp_size) {
        ComputeBatchId(stream, unique_hashes_batch_id, num_voxels, batch_hash);
    }

    std::pair<int64_t*, size_t> unique_batches =
            mem_temp.Alloc<int64_t>(batch_size);
    std::pair<int64_t*, size_t> unique_batches_count =
            mem_temp.Alloc<int64_t>(batch_size);
    int64_t num_batches = 0;  // Store non empty batches

    // Convert batch_id to counts (array of num_voxels per batch)
    {
        std::pair<void*, size_t> encode_temp(nullptr, 0);
        std::pair<int64_t*, size_t> num_batches_mem =
                mem_temp.Alloc<int64_t>(1);

        cub::DeviceRunLengthEncode::Encode(
                encode_temp.first, encode_temp.second, unique_hashes_batch_id,
                unique_batches.first, unique_batches_count.first,
                num_batches_mem.first, num_voxels, stream);
        encode_temp = mem_temp.Alloc(encode_temp.second);
        if (!get_temp_size) {
            cub::DeviceRunLengthEncode::Encode(
                    encode_temp.first, encode_temp.second,
                    unique_hashes_batch_id, unique_batches.first,
                    unique_batches_count.first, num_batches_mem.first,
                    num_voxels, stream);

            // get the number of non empty batches.
            cudaMemcpyAsync(&num_batches, num_batches_mem.first,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            // wait for the async copies
            while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
            }
        }
        mem_temp.Free(encode_temp);
    }

    // Insert count(0) for empty batches
    std::pair<int64_t*, size_t> num_voxels_per_batch =
            mem_temp.Alloc<int64_t>(batch_size);
    if (!get_temp_size) {
        cudaMemset(num_voxels_per_batch.first, 0, batch_size * sizeof(int64_t));
        ComputeVoxelPerBatch(stream, num_voxels_per_batch.first,
                             unique_batches_count.first, unique_batches.first,
                             num_batches);
    }

    std::pair<int64_t*, size_t> num_voxels_prefix_sum(unique_batches.first,
                                                      batch_size);

    // compute the prefix sum for number of voxels per batch
    // gives starting voxel index for each batch
    // used only when voxel count exceeds max_voxels
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);

        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                num_voxels_per_batch.first, num_voxels_prefix_sum.first,
                num_voxels_per_batch.second, stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);
        if (!get_temp_size) {
            if (num_voxels > max_voxels) {
                cub::DeviceScan::InclusiveSum(
                        inclusive_scan_temp.first, inclusive_scan_temp.second,
                        num_voxels_per_batch.first, num_voxels_prefix_sum.first,
                        num_voxels_per_batch.second, stream);
            }
        }
        mem_temp.Free(inclusive_scan_temp);
    }

    // Limit the number of voxels per batch to max_voxels
    if (!get_temp_size) {
        if (num_voxels >= max_voxels)
            LimitCounts(stream, num_voxels_per_batch.first, batch_size,
                        max_voxels);
    }

    // Prefix sum of limited counts to get batch splits.
    int64_t* out_batch_splits = nullptr;
    if (!get_temp_size) {
        output_allocator.AllocVoxelBatchSplits(&out_batch_splits,
                                               batch_size + 1);
        cudaMemsetAsync(out_batch_splits, 0, sizeof(int64_t), stream);
    }

    // Prefix sum of counts to get batch splits
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);

        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                num_voxels_per_batch.first, out_batch_splits + 1,
                num_voxels_per_batch.second, stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);

        if (!get_temp_size) {
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    num_voxels_per_batch.first, out_batch_splits + 1,
                    batch_size, stream);
        }
        mem_temp.Free(inclusive_scan_temp);
    }

    // num_valid_voxels excludes voxels exceeding max_voxels
    int64_t num_valid_voxels = num_points;
    if (!get_temp_size) {
        // get the number of valid voxels.
        cudaMemcpyAsync(&num_valid_voxels, out_batch_splits + batch_size,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        // wait for the async copies
        while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
        }
    }

    // start_idx stores starting index of each valid voxel.
    // points_count stores number of valid points in respective voxel.
    std::pair<int64_t*, size_t> start_idx(indices_batches.first,
                                          num_valid_voxels);
    std::pair<int64_t*, size_t> points_count =
            mem_temp.Alloc<int64_t>(num_valid_voxels);

    if (!get_temp_size) {
        if (num_voxels <= max_voxels) {
            // starting index and points_count will be same as
            // unique_hashes_count_prefix_sum and unique_hashes_count when voxel
            // count doesn't exceeds max_voxels
            cudaMemsetAsync(start_idx.first, 0, sizeof(int64_t), stream);
            cudaMemcpyAsync(start_idx.first + 1,
                            unique_hashes_count_prefix_sum.first,
                            (num_voxels - 1) * sizeof(int64_t),
                            cudaMemcpyDeviceToDevice, stream);
            mem_temp.Free(points_count);
            points_count.first = unique_hashes_count.first;
        } else {
            ComputeStartIdx(stream, start_idx.first, points_count.first,
                            num_voxels_prefix_sum.first,
                            unique_hashes_count_prefix_sum.first,
                            out_batch_splits, batch_size, max_voxels,
                            max_points_per_voxel);
        }
    }

    int64_t* out_voxel_row_splits = nullptr;
    if (!get_temp_size) {
        output_allocator.AllocVoxelPointRowSplits(&out_voxel_row_splits,
                                                  num_valid_voxels + 1);
    }

    if (!get_temp_size) {
        // set first element to 0
        cudaMemsetAsync(out_voxel_row_splits, 0, sizeof(int64_t), stream);
    }
    {
        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);

        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                points_count.first, out_voxel_row_splits + 1, num_valid_voxels,
                stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);
        if (!get_temp_size) {
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    points_count.first, out_voxel_row_splits + 1,
                    num_valid_voxels, stream);
        }
        mem_temp.Free(inclusive_scan_temp);
    }

    if (get_temp_size) {
        // return the memory peak as the required temporary memory size.
        temp_size = mem_temp.MaxUsed();
        return;
    }

    int32_t* out_voxel_coords = nullptr;
    output_allocator.AllocVoxelCoords(&out_voxel_coords, num_valid_voxels,
                                      NDIM);
    ComputeVoxelCoords(stream, out_voxel_coords, points,
                       point_indices_dbuf.Current(), start_idx.first,
                       points_range_min_vec, inv_voxel_size, num_valid_voxels);

    int64_t num_valid_points = 0;
    {
        cudaMemcpyAsync(&num_valid_points,
                        out_voxel_row_splits + num_valid_voxels,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        // wait for the async copies
        while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
        }
    }
    int64_t* out_point_indices = nullptr;
    output_allocator.AllocVoxelPointIndices(&out_point_indices,
                                            num_valid_points);
    CopyPointIndices(stream, out_point_indices, point_indices_dbuf.Current(),
                     start_idx.first, out_voxel_row_splits + 1,
                     num_valid_voxels);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
