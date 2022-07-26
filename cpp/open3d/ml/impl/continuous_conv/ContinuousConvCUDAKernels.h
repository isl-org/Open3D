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

#include <vector>

#include "open3d/ml/impl/continuous_conv/CoordinateTransformation.cuh"

namespace open3d {
namespace ml {
namespace impl {

/// Copies and transforms the features to a column, which can be multiplied
/// with the filter matrix.
///
/// \tparam TFeat    Type for the features
/// \tparam TOut     Type for the output features
/// \tparam TReal    Type for point positions and extents
/// \tparam TIndex   Type for neighbor indexing
///
/// \param columns    Output array with shape
///        [num_out, spatial filter dims, in_channels].
///
/// \param in_channels    Number of input channels.
///
/// \param begin_idx    Index of the first output point to process.
///
/// \param end_idx    Index after the last output point to process.
///
/// \param num_out    The number of output points.
///
/// \param out_positions    The positions of the output points. The shape is
///        [num_out, 3].
///
/// \param num_inp    The number of input points.
///
/// \param inp_positions    The positions of the input points. The shape is
///        [num_inp, 3].
///
/// \param inp_features    The input features with shape
///        [num_inp, in_channels].
///
/// \param inp_importance    Optional importance for each input point with
///        shape [num_inp]. Set to null to disable.
///
/// \param neighbors_index_size    The size of the neighbors_index array.
///
/// \param neighbors_index    The array with lists of neighbors for each output
///        point. The start and end of each sublist is defined by
///        \p neighbors_row_splits.
///
/// \param neighbors_importance    Optional importance for each entry in
///        \p neighbors_index. Set to null to disable.
///
/// \param neighbors_row_splits   The prefix sum which defines the start
///        and end of the sublists in \p neighbors_index. The size of the
///        array is \p num_out + 1.
///
/// \param extents    The spatial extents of the filter in coordinate units.
///        extents can be a scalar or a 1D array of shape [num_out] or a
///        2D array of shape [num_out,3]. The shape depends on
///        \p individual_extent and \p isotropic_extent.
///
/// \param offsets    A single 3D vector used in the filter coordinate
///        computation. The shape is [3].
///
/// \param filter_dims    The spatial filter size in voxels. (The filter
///        resolution) with shape [3].
///
/// \param interpolation    The interpolation mode. Either LINEAR or
///        NEAREST_NEIGHBOR.
///
/// \param coordinate_mapping    The coordinate mapping function. One of
///        IDENTITY, BALL_TO_CUBE_RADIAL, BALL_TO_CUBE_VOLUME_PRESERVING.
///
/// \param align_corners    If true then the voxel centers of the outer voxels
///        of the filter array are mapped to the boundary of the filter shape.
///        If false then the boundary of the filter array is mapped to the
///        boundary of the filter shape.
///
/// \param individual_extent    If true each output point has an individual
///        extent.
///
/// \param isotropic_extent    If true each then the extent is isotropic for
///        each output point.
///
/// \param normalize    If true then the result is normalized either by the
///        number of points (neighbors_importance is null) or by the sum of
///        the respective values in neighbors_importance.
///
template <class TFeat, class TReal, class TIndex>
void FillColumn(const cudaStream_t& stream,
                TFeat* columns,
                int in_channels,
                TIndex begin_idx,
                TIndex end_idx,
                TIndex num_out,
                const TReal* const __restrict__ out_positions,
                TIndex num_inp,
                const TReal* const __restrict__ inp_positions,
                const TFeat* const __restrict__ inp_features,
                const TFeat* const __restrict__ inp_importance,
                size_t neighbors_index_size,
                const TIndex* const __restrict__ neighbors_index,
                const TFeat* const __restrict__ neighbors_importance,
                const int64_t* const __restrict__ neighbors_row_splits,
                const TReal* const __restrict__ extents,
                const TReal* const __restrict__ offsets,
                const std::vector<int>& filter_dims,
                InterpolationMode interpolation,
                CoordinateMapping coordinate_mapping,
                bool align_corners,
                bool individual_extent,
                bool isotropic_extent,
                bool normalize);

template <class TFeat, class TReal, class TIndex>
void FillColumnTranspose(
        const cudaStream_t& stream,
        TFeat* columns,
        int in_channels,
        TIndex begin_idx,
        TIndex end_idx,
        TIndex num_out,
        const TReal* const __restrict__ out_positions,
        TIndex num_inp,
        const TReal* const __restrict__ inp_positions,
        const TFeat* const __restrict__ inp_features,
        const TFeat* const __restrict__ inp_neighbors_importance_sum,
        const int64_t* const __restrict__ inp_neighbors_prefix_sum,
        size_t neighbors_index_size,
        const TIndex* const __restrict__ neighbors_index,
        const TFeat* const __restrict__ neighbors_importance,
        const int64_t* const __restrict__ neighbors_row_splits,
        const TReal* const __restrict__ extents,
        const TReal* const __restrict__ offsets,
        const std::vector<int>& filter_dims,
        InterpolationMode interpolation,
        CoordinateMapping coordinate_mapping,
        bool align_corners,
        bool individual_extent,
        bool isotropic_extent,
        bool normalize);

/// Multiplies each column with a scalar in-place.
///
/// \param col_major_matrix    Matrix with shape [rows, cols] in column major
///        storage order.
///
/// \param vector    A vector with shape [cols].
///
template <class T>
void MultiplyColumns(const cudaStream_t& stream,
                     size_t rows,
                     size_t cols,
                     T* __restrict__ col_major_matrix,
                     const T* const __restrict__ vector);

/// Multiplies each column with a scalar.
///
/// \param out_ptr    Output pointer
///
/// \param col_major_matrix    Matrix with shape [rows, cols] in column major
///        storage order.
///
/// \param vector    A vector with shape [cols].
///
template <class T>
void MultiplyAndCopyColumns(const cudaStream_t& stream,
                            size_t rows,
                            size_t cols,
                            T* __restrict__ out_ptr,
                            const T* const __restrict__ col_major_matrix,
                            const T* const __restrict__ vector);

}  // namespace impl
}  // namespace ml
}  // namespace open3d
