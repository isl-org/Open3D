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

#include <tbb/parallel_for.h>

#include "open3d/ml/impl/continuous_conv/CoordinateTransformation.h"

namespace open3d {
namespace ml {
namespace impl {

/// Implementation of CConvComputeFeatures with template parameters for
/// configuration.
template <class TReal,
          class TIndex,
          InterpolationMode INTERPOLATION,
          CoordinateMapping MAPPING,
          bool ALIGN_CORNERS,
          bool INDIVIDUAL_EXTENT,
          bool ISOTROPIC_EXTENT,
          bool NORMALIZE>
void _CConvTransposeComputeFeaturesCPU(
        TReal* out_features,
        const std::vector<int>& filter_dims,
        const TReal* filter,
        size_t num_out,
        const TReal* out_positions,
        const TReal* out_importance,
        size_t num_inp,
        const TReal* inp_positions,
        const TReal* inp_features,
        const TReal* inp_neighbors_importance_sum,
        const int64_t* inp_neighbors_row_splits,
        size_t neighbors_index_size,
        const TIndex* neighbor_index,
        const TReal* neighbor_importance,
        const int64_t* neighbors_row_splits,
        const TReal* extents,
        const TReal* offsets) {
    const bool NEIGHBOR_IMPORTANCE = inp_neighbors_importance_sum;
    const int VECSIZE = 32;
    typedef Eigen::Array<TReal, VECSIZE, 1> Vec_t;
    typedef InterpolationVec<TReal, VECSIZE, INTERPOLATION> InterpolationVec_t;
    InterpolationVec_t interpolation;

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int spatial_filter_size = 1;
    for (int i = 0; i < 3; ++i) spatial_filter_size *= filter_dims[i];
    Eigen::Array<int, 3, 1> filter_size_xyz(filter_dims[2], filter_dims[1],
                                            filter_dims[0]);

    memset(out_features, 0, sizeof(TReal) * num_out * out_channels);

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_out, 32),
            [&](const tbb::blocked_range<size_t>& r) {

                int range_length = r.end() - r.begin();

                Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> B(
                        in_channels * spatial_filter_size, range_length);
                B.setZero();

                typedef Eigen::Array<TReal, VECSIZE, Eigen::Dynamic> Matrix;
                Matrix infeat(VECSIZE, in_channels);

                Eigen::Array<TReal, 3, 1> offsets_(offsets[0], offsets[1],
                                                   offsets[2]);

                Eigen::Array<TReal, VECSIZE, 3> inv_extents;
                if (INDIVIDUAL_EXTENT == false) {
                    if (ISOTROPIC_EXTENT) {
                        inv_extents = 1 / extents[0];
                    } else {
                        inv_extents.col(0) = 1 / extents[0];
                        inv_extents.col(1) = 1 / extents[1];
                        inv_extents.col(2) = 1 / extents[2];
                    }
                }

                for (size_t out_idx = r.begin(); out_idx != r.end();
                     ++out_idx) {
                    const int out_col = out_idx - r.begin();
                    const size_t neighbor_start = neighbors_row_splits[out_idx];
                    const size_t neighbor_end =
                            (out_idx + 1 < num_out
                                     ? neighbors_row_splits[out_idx + 1]
                                     : neighbors_index_size);

                    typename InterpolationVec_t::Weight_t interp_weights;
                    typename InterpolationVec_t::Idx_t interp_indices;

                    int vec_valid_count = 0;
                    Vec_t x, y, z;

                    // set to zero to avoid problems with vectors with less than
                    // VECSIZE valid entries
                    x.setZero();
                    y.setZero();
                    z.setZero();
                    for (size_t n = neighbor_start; n < neighbor_end; ++n) {
                        const size_t inp_idx = neighbor_index[n];

                        const int i = vec_valid_count;
                        x(i) = out_positions[out_idx * 3 + 0] -
                               inp_positions[inp_idx * 3 + 0];
                        y(i) = out_positions[out_idx * 3 + 1] -
                               inp_positions[inp_idx * 3 + 1];
                        z(i) = out_positions[out_idx * 3 + 2] -
                               inp_positions[inp_idx * 3 + 2];

                        if (INDIVIDUAL_EXTENT) {
                            if (ISOTROPIC_EXTENT) {
                                inv_extents.row(i) = 1 / extents[inp_idx];
                            } else {
                                inv_extents(i, 0) =
                                        1 / extents[3 * inp_idx + 0];
                                inv_extents(i, 1) =
                                        1 / extents[3 * inp_idx + 1];
                                inv_extents(i, 2) =
                                        1 / extents[3 * inp_idx + 2];
                            }
                        }

                        TReal n_importance = NEIGHBOR_IMPORTANCE
                                                     ? neighbor_importance[n]
                                                     : 1;
                        for (int ic = 0; ic < in_channels; ++ic)
                            infeat(i, ic) =
                                    inp_features[inp_idx * in_channels + ic] *
                                    n_importance;

                        if (NORMALIZE) {
                            TReal normalizer = 1;
                            if (NEIGHBOR_IMPORTANCE) {
                                if (inp_neighbors_importance_sum[inp_idx] != 0)
                                    normalizer /= inp_neighbors_importance_sum
                                            [inp_idx];
                            } else {
                                size_t num_inp_neighbors;
                                const size_t inp_neighbor_start =
                                        inp_neighbors_row_splits[inp_idx];
                                const size_t inp_neighbor_end =
                                        inp_neighbors_row_splits[inp_idx + 1];
                                num_inp_neighbors =
                                        inp_neighbor_end - inp_neighbor_start;
                                if (num_inp_neighbors > 0)
                                    normalizer /= num_inp_neighbors;
                            }
                            for (int ic = 0; ic < in_channels; ++ic)
                                infeat(i, ic) *= normalizer;
                        }

                        ++vec_valid_count;
                        if (vec_valid_count == VECSIZE ||
                            n + 1 == neighbor_end) {
                            ComputeFilterCoordinates<ALIGN_CORNERS, MAPPING>(
                                    x, y, z, filter_size_xyz, inv_extents,
                                    offsets_);
                            interpolation.Interpolate(
                                    interp_weights, interp_indices, x, y, z,
                                    filter_size_xyz, in_channels);
                            for (int k = 0; k < vec_valid_count; ++k) {
                                for (int j = 0; j < InterpolationVec_t::Size();
                                     ++j) {
                                    for (int ic = 0; ic < in_channels; ++ic)
                                        B(interp_indices(j, k) + ic, out_col) +=
                                                interp_weights(j, k) *
                                                infeat(k, ic);
                                }
                            }
                            vec_valid_count = 0;
                        }
                    }

                }  // out_idx

                Eigen::Map<const Eigen::Matrix<TReal, Eigen::Dynamic,
                                               Eigen::Dynamic>>
                        A(filter, out_channels,
                          spatial_filter_size * in_channels);
                Eigen::Map<Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic>>
                        C(out_features + (r.begin() * out_channels),
                          out_channels, range_length);

                C = A * B;
                if (out_importance) {
                    for (int i = 0; i < range_length; ++i)
                        C.col(i) *= out_importance[r.begin() + i];
                }

            });
}

/// Computes the output features of a transpose continuous convolution.
///
/// \param out_features    Output array for the computed features with shape
///        [num_out, out channels]
///
/// \param filter_dims    The sizes of the filter dimensions. The size of
///        filter_dims must be 5. The order is
///        [depth, height, width, inp channels, out channels].
///
/// \param filter    Pointer to the filter values.
///
/// \param num_out    The number of output points.
///
/// \param out_positions    The positions of the output points. The shape is
///        [num_out, 3].
///
/// \param out_importance    Optional importance for each output point with
///        shape [num_out]. Set to null to disable.
///
/// \param num_inp    The number of input points.
///
/// \param inp_positions    The positions of the input points. The shape is
///        [num_inp, 3].
///
/// \param inp_features    The input features with shape
///        [num_inp, in_channels].
///
/// \param inp_neighbors_importance_sum    The sum of the neighbors_importance
///        values for each input with shape [num_inp].
///
/// \param inp_neighbors_row_splits   The prefix sum which defines the start
///        and end of the sublists in \p inp_neighbors_index. The size of the
///        array is \p num_inp + 1.
///
/// \param neighbors_index_size    The size of the neighbors_index array.
///
/// \param neighbors_index    The array with lists of neighbors for each
///        output point. The start and end of each sublist is defined by
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
template <class TReal, class TIndex>
void CConvTransposeComputeFeaturesCPU(TReal* out_features,
                                      const std::vector<int>& filter_dims,
                                      const TReal* filter,
                                      size_t num_out,
                                      const TReal* out_positions,
                                      const TReal* out_importance,
                                      size_t num_inp,
                                      const TReal* inp_positions,
                                      const TReal* inp_features,
                                      const TReal* inp_neighbors_importance_sum,
                                      const int64_t* inp_neighbors_row_splits,
                                      size_t neighbors_index_size,
                                      const TIndex* neighbor_index,
                                      const TReal* neighbor_importance,
                                      const int64_t* neighbors_row_splits,
                                      const TReal* extents,
                                      const TReal* offsets,
                                      InterpolationMode interpolation,
                                      CoordinateMapping coordinate_mapping,
                                      bool align_corners,
                                      bool individual_extent,
                                      bool isotropic_extent,
                                      bool normalize) {
#define FN_PARAMETERS                                                          \
    out_features, filter_dims, filter, num_out, out_positions, out_importance, \
            num_inp, inp_positions, inp_features,                              \
            inp_neighbors_importance_sum, inp_neighbors_row_splits,            \
            neighbors_index_size, neighbor_index, neighbor_importance,         \
            neighbors_row_splits, extents, offsets

#define CALL_TEMPLATE(INTERPOLATION, MAPPING, ALIGN_CORNERS,               \
                      INDIVIDUAL_EXTENT, ISOTROPIC_EXTENT, NORMALIZE)      \
    if (INTERPOLATION == interpolation && MAPPING == coordinate_mapping && \
        ALIGN_CORNERS == align_corners &&                                  \
        INDIVIDUAL_EXTENT == individual_extent &&                          \
        ISOTROPIC_EXTENT == isotropic_extent && NORMALIZE == normalize)    \
        _CConvTransposeComputeFeaturesCPU<                                 \
                TReal, TIndex, INTERPOLATION, MAPPING, ALIGN_CORNERS,      \
                INDIVIDUAL_EXTENT, ISOTROPIC_EXTENT, NORMALIZE>(           \
                FN_PARAMETERS);

#define CALL_TEMPLATE2(INTERPOLATION, MAPPING)                       \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, true, true, true)    \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, true, true, false)   \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, true, false, true)   \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, true, false, false)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, false, true, true)   \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, false, true, false)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, false, false, true)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, true, false, false, false) \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, true, true, true)   \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, true, true, false)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, true, false, true)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, true, false, false) \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, false, true, true)  \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, false, true, false) \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, false, false, true) \
    CALL_TEMPLATE(INTERPOLATION, MAPPING, false, false, false, false)

#define CALL_TEMPLATE3(INTERPOLATION)                                     \
    CALL_TEMPLATE2(INTERPOLATION, CoordinateMapping::BALL_TO_CUBE_RADIAL) \
    CALL_TEMPLATE2(INTERPOLATION,                                         \
                   CoordinateMapping::BALL_TO_CUBE_VOLUME_PRESERVING)     \
    CALL_TEMPLATE2(INTERPOLATION, CoordinateMapping::IDENTITY)

#define CALL_TEMPLATE4                               \
    CALL_TEMPLATE3(InterpolationMode::LINEAR)        \
    CALL_TEMPLATE3(InterpolationMode::LINEAR_BORDER) \
    CALL_TEMPLATE3(InterpolationMode::NEAREST_NEIGHBOR)

    CALL_TEMPLATE4

#undef CALL_TEMPLATE
#undef CALL_TEMPLATE2
#undef CALL_TEMPLATE3
#undef CALL_TEMPLATE4

#undef FN_PARAMETERS
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
