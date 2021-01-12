// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

template <class T, int NDIM>
__global__ void ComputeHashKernel(
        int64_t* __restrict__ hashes,
        int64_t num_points,
        const T* const __restrict__ points,
        const open3d::utility::MiniVec<T, NDIM> points_range_min_vec,
        const open3d::utility::MiniVec<T, NDIM> points_range_max_vec,
        const open3d::utility::MiniVec<T, NDIM> inv_voxel_size,
        const open3d::utility::MiniVec<int64_t, NDIM> strides,
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
        hashes[linear_idx] = h;  // add 1 and use 0 as invalid
    } else {
        hashes[linear_idx] = invalid_hash;  // 0 means invalid
    }
}

/// This function computes the a hash (linear index+1) for each point.
/// Points outside the range will get a specific hash value.
///
/// \tparam T    The floating point type for the points
/// \tparam NDIM    The number of dimensions, e.g., 3.
///
/// \param hashes    The output vector with the hashes/linear indexes.
/// \param num_points    The number of points.
/// \param points    The array with the point coordinates. The shape is
///        [num_points,NDIM] and the storage order is row-major.
/// \param points_range_min_vec    The minimum range for a point to be valid.
/// \param points_range_max_vec    The maximum range for a point to be valid.
/// \param inv_voxel_size    The reciprocal of the voxel edge lengths in each
///        dimension
/// \param strides    The strides for computing the linear index.
/// \param invalid_hash    The value to use for points outside the range.
template <class T, int NDIM>
void ComputeHash(const cudaStream_t& stream,
                 int64_t* hashes,
                 int64_t num_points,
                 const T* const points,
                 const MiniVec<T, NDIM> points_range_min_vec,
                 const MiniVec<T, NDIM> points_range_max_vec,
                 const MiniVec<T, NDIM> inv_voxel_size,
                 const MiniVec<int64_t, NDIM> strides,
                 const int64_t invalid_hash) {
    if (num_points) {
        const int BLOCKSIZE = 128;
        dim3 block(BLOCKSIZE, 1, 1);
        dim3 grid;
        grid.y = std::ceil(std::cbrt(num_points));
        grid.z = grid.y;
        grid.x = DivUp(num_points, int64_t(grid.z) * grid.y * block.x);
        ComputeHashKernel<T, NDIM><<<grid, block, 0, stream>>>(
                hashes, num_points, points, points_range_min_vec,
                points_range_max_vec, inv_voxel_size, strides, invalid_hash);
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

    int64_t point_idx;
    if (0 == linear_idx) {
        point_idx = point_indices[0];
    } else {
        point_idx = point_indices[prefix_sum[linear_idx - 1]];
    }

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

    int64_t in_idx;
    if (0 == linear_idx) {
        in_idx = 0;
    } else {
        in_idx = prefix_sum_in[linear_idx - 1];
    }

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
/// \param voxel_size    The edge lenghts of the voxel. The shape is
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
///         size), and AllocVoxelPointRowSplits(int64_t** ptr, int64_t
///         size). All functions should allocate memory and return a
///         pointer to that memory in ptr. The argments size, rows, and
///         cols define the size of the array as the number of elements.
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
    const int64_t invalid_hash = strides[NDIM - 1] * extents[NDIM - 1];

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
        ComputeHash(stream, hashes.first, num_points, points,
                    points_range_min_vec, points_range_max_vec, inv_voxel_size,
                    strides, invalid_hash);
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
    num_voxels = std::min(int64_t(num_voxels), max_voxels);

    // reuse the hashes buffer
    std::pair<int64_t*, size_t> unique_hashes_count_prefix_sum(
            hashes_dbuf.Current(), hashes.second);

    // compute the prefix sum for unique_hashes_count
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

    int64_t* out_voxel_row_splits = nullptr;
    if (!get_temp_size) {
        output_allocator.AllocVoxelPointRowSplits(&out_voxel_row_splits,
                                                  num_voxels + 1);
    }

    if (!get_temp_size) {
        // set first element to 0
        cudaMemsetAsync(out_voxel_row_splits, 0, sizeof(int64_t), stream);
    }

    if (max_points_per_voxel < num_points) {
        // Limit the number of output points to max_points_per_voxel by
        // limiting the unique_hashes_count.
        if (!get_temp_size) {
            LimitCounts(stream, unique_hashes_count.first, num_voxels,
                        max_points_per_voxel);
        }

        std::pair<void*, size_t> inclusive_scan_temp(nullptr, 0);

        cub::DeviceScan::InclusiveSum(
                inclusive_scan_temp.first, inclusive_scan_temp.second,
                unique_hashes_count.first, out_voxel_row_splits + 1, num_voxels,
                stream);

        inclusive_scan_temp = mem_temp.Alloc(inclusive_scan_temp.second);
        if (!get_temp_size) {
            cub::DeviceScan::InclusiveSum(
                    inclusive_scan_temp.first, inclusive_scan_temp.second,
                    unique_hashes_count.first, out_voxel_row_splits + 1,
                    num_voxels, stream);
        }
        mem_temp.Free(inclusive_scan_temp);

    } else {
        if (!get_temp_size) {
            cudaMemcpyAsync(out_voxel_row_splits + 1,
                            unique_hashes_count_prefix_sum.first,
                            sizeof(int64_t) * num_voxels,
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    if (get_temp_size) {
        // return the memory peak as the required temporary memory size.
        temp_size = mem_temp.MaxUsed();
        return;
    }

    int32_t* out_voxel_coords = nullptr;
    output_allocator.AllocVoxelCoords(&out_voxel_coords, num_voxels, NDIM);
    ComputeVoxelCoords(stream, out_voxel_coords, points,
                       point_indices_dbuf.Current(),
                       unique_hashes_count_prefix_sum.first,
                       points_range_min_vec, inv_voxel_size, num_voxels);

    int64_t num_valid_points = 0;
    {
        cudaMemcpyAsync(&num_valid_points, out_voxel_row_splits + num_voxels,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        // wait for the async copies
        while (cudaErrorNotReady == cudaStreamQuery(stream)) { /*empty*/
        }
    }
    int64_t* out_point_indices = nullptr;
    output_allocator.AllocVoxelPointIndices(&out_point_indices,
                                            num_valid_points);
    CopyPointIndices(stream, out_point_indices, point_indices_dbuf.Current(),
                     unique_hashes_count_prefix_sum.first,
                     out_voxel_row_splits + 1, num_voxels);
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
