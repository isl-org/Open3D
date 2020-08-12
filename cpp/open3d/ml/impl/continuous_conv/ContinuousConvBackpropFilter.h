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

#include <mutex>

#include "open3d/ml/impl/continuous_conv/CoordinateTransformation.h"

namespace open3d {
namespace ml {
namespace impl {

// Implementation of CConvBackropFilterCPU
template <class TReal,
          class TIndex,
          InterpolationMode INTERPOLATION,
          CoordinateMapping MAPPING,
          bool ALIGN_CORNERS,
          bool INDIVIDUAL_EXTENT,
          bool ISOTROPIC_EXTENT,
          bool POINT_IMPORTANCE>
void _CConvBackropFilterCPU(TReal* filter_backprop,
                            const std::vector<int>& filter_dims,
                            size_t num_out,
                            const TReal* out_positions,
                            size_t num_inp,
                            const TReal* inp_positions,
                            const TReal* inp_features,
                            const TReal* inp_importance,
                            size_t num_indices,
                            const TIndex* neighbors_index,
                            const TReal* neighbors_importance,
                            const int64_t* neighbors_row_splits,
                            const TReal* extents,
                            const TReal* offsets,
                            const TReal* out_features_gradient,
                            bool normalize) {
    const bool NEIGHBOR_IMPORTANCE = neighbors_importance;
    const int VECSIZE = 32;
    typedef Eigen::Array<TReal, VECSIZE, 1> Vec_t;
    typedef InterpolationVec<TReal, VECSIZE, INTERPOLATION> InterpolationVec_t;
    InterpolationVec_t interpolation;

    const int in_channels = filter_dims[filter_dims.size() - 2];
    const int out_channels = filter_dims[filter_dims.size() - 1];

    int spatial_filter_size = 1;
    for (int i = 0; i < 3; ++i) spatial_filter_size *= filter_dims[i];
    const int total_filter_size =
            spatial_filter_size * in_channels * out_channels;
    Eigen::Array<int, 3, 1> filter_size_xyz(filter_dims[2], filter_dims[1],
                                            filter_dims[0]);

    memset(filter_backprop, 0, sizeof(TReal) * total_filter_size);
    std::mutex filter_backprop_mutex;

    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_out, 32),
            [&](const tbb::blocked_range<size_t>& r) {
                int range_length = r.end() - r.begin();

                Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> B(
                        in_channels * spatial_filter_size, range_length);
                B.setZero();
                Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> C(
                        out_channels, range_length);

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
                            neighbors_row_splits[out_idx + 1];
                    TReal normalizer = 0;

                    if (INDIVIDUAL_EXTENT) {
                        if (ISOTROPIC_EXTENT) {
                            inv_extents = 1 / extents[out_idx];
                        } else {
                            inv_extents.col(0) = 1 / extents[3 * out_idx + 0];
                            inv_extents.col(1) = 1 / extents[3 * out_idx + 1];
                            inv_extents.col(2) = 1 / extents[3 * out_idx + 2];
                        }
                    }

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
                        const size_t inp_idx = neighbors_index[n];
                        const int i = vec_valid_count;
                        x(i) = inp_positions[inp_idx * 3 + 0] -
                               out_positions[out_idx * 3 + 0];
                        y(i) = inp_positions[inp_idx * 3 + 1] -
                               out_positions[out_idx * 3 + 1];
                        z(i) = inp_positions[inp_idx * 3 + 2] -
                               out_positions[out_idx * 3 + 2];

                        const TReal n_importance =
                                (NEIGHBOR_IMPORTANCE ? neighbors_importance[n]
                                                     : 1);
                        normalizer += n_importance;

                        for (int ic = 0; ic < in_channels; ++ic)
                            infeat(i, ic) =
                                    inp_features[inp_idx * in_channels + ic];

                        TReal importance = 1;
                        if (POINT_IMPORTANCE)
                            importance = inp_importance[inp_idx];
                        if (NEIGHBOR_IMPORTANCE) importance *= n_importance;

                        if (POINT_IMPORTANCE || NEIGHBOR_IMPORTANCE) {
                            for (int ic = 0; ic < in_channels; ++ic)
                                infeat(i, ic) *= importance;
                        }

                        ++vec_valid_count;
                        if (vec_valid_count == VECSIZE) {
                            ComputeFilterCoordinates<ALIGN_CORNERS, MAPPING>(
                                    x, y, z, filter_size_xyz, inv_extents,
                                    offsets_);
                            interpolation.Interpolate(
                                    interp_weights, interp_indices, x, y, z,
                                    filter_size_xyz, in_channels);
                            for (int k = 0; k < VECSIZE; ++k)
                                for (int j = 0; j < InterpolationVec_t::Size();
                                     ++j) {
                                    for (int ic = 0; ic < in_channels; ++ic)
                                        B(interp_indices(j, k) + ic, out_col) +=
                                                interp_weights(j, k) *
                                                infeat(k, ic);
                                }
                            vec_valid_count = 0;
                        }
                    }
                    if (vec_valid_count) {
                        ComputeFilterCoordinates<ALIGN_CORNERS, MAPPING>(
                                x, y, z, filter_size_xyz, inv_extents,
                                offsets_);
                        interpolation.Interpolate(interp_weights,
                                                  interp_indices, x, y, z,
                                                  filter_size_xyz, in_channels);
                        for (int k = 0; k < vec_valid_count; ++k)
                            for (int j = 0; j < InterpolationVec_t::Size();
                                 ++j) {
                                for (int ic = 0; ic < in_channels; ++ic)
                                    B(interp_indices(j, k) + ic, out_col) +=
                                            interp_weights(j, k) *
                                            infeat(k, ic);
                            }
                    }

                    C.col(out_col) = Eigen::Map<
                            const Eigen::Array<TReal, Eigen::Dynamic, 1>>(
                            out_features_gradient + out_idx * out_channels,
                            out_channels, 1);

                    if (normalize && normalizer != 0)
                        C.col(out_col) /= normalizer;

                }  // out_idx

                Eigen::Matrix<TReal, Eigen::Dynamic, Eigen::Dynamic> A(
                        out_channels, spatial_filter_size * in_channels);

                A = C * B.transpose();

                {
                    std::lock_guard<std::mutex> lock(filter_backprop_mutex);
                    int linear_i = 0;
                    for (int j = 0; j < spatial_filter_size * in_channels; ++j)
                        for (int i = 0; i < out_channels; ++i, ++linear_i) {
                            filter_backprop[linear_i] += A(i, j);
                        }
                }
            });
}

/// Computes the backprop for the filter of a continuous convolution.
///
/// \param temp    Pointer to temporary memory. If nullptr then the required
///        size of temporary memory will be written to \p temp_size and no
///        work is done. This function can make use of more memory and
///        returns the maximum size that can be used in max_temp_size.
///
/// \param temp_size    The size of the temporary memory in bytes. This is
///        used as an output if temp is nullptr and returns the minimum temp
///        size required.
///
/// \param max_temp_size    This is used as an output if temp is nullptr and
///        returns the maximum temp size that can be used.
///
/// \param texture_alignment    The texture alignment in bytes. This is used
///        for allocating segments within the temporary memory.
///
/// \param filter_backrop    Output array for the computed filter gradient
///        with shape [depth,height,witdth, inp channels, out channels]
///
/// \param filter_dims    The sizes of the filter dimensions. The size of
///        filter_dims must be 5. The order is
///        [depth, height, width, inp channels, out channels].
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
/// \param normalize    If true then the output features are normalized either
///        by the number of points (neighbors_importance is null) or by the sum
///        of the respective values in neighbors_importance.
///
template <class TReal, class TIndex>
void CConvBackpropFilterCPU(TReal* filter_backprop,
                            const std::vector<int>& filter_dims,
                            size_t num_out,
                            const TReal* out_positions,
                            size_t num_inp,
                            const TReal* inp_positions,
                            const TReal* inp_features,
                            const TReal* inp_importance,
                            size_t num_indices,
                            const TIndex* neighbors_index,
                            const TReal* neighbors_importance,
                            const int64_t* neighbors_row_splits,
                            const TReal* extents,
                            const TReal* offsets,
                            const TReal* out_features_gradient,
                            InterpolationMode interpolation,
                            CoordinateMapping coordinate_mapping,
                            bool align_corners,
                            bool individual_extent,
                            bool isotropic_extent,
                            bool normalize) {
    bool has_importance = inp_importance;

#define FN_PARAMETERS                                                    \
    filter_backprop, filter_dims, num_out, out_positions, num_inp,       \
            inp_positions, inp_features, inp_importance, num_indices,    \
            neighbors_index, neighbors_importance, neighbors_row_splits, \
            extents, offsets, out_features_gradient, normalize

#define CALL_TEMPLATE(INTERPOLATION, MAPPING, ALIGN_CORNERS,               \
                      INDIVIDUAL_EXTENT, ISOTROPIC_EXTENT, HAS_IMPORTANCE) \
    if (INTERPOLATION == interpolation && MAPPING == coordinate_mapping && \
        ALIGN_CORNERS == align_corners &&                                  \
        INDIVIDUAL_EXTENT == individual_extent &&                          \
        ISOTROPIC_EXTENT == isotropic_extent &&                            \
        HAS_IMPORTANCE == has_importance)                                  \
        _CConvBackropFilterCPU<TReal, TIndex, INTERPOLATION, MAPPING,      \
                               ALIGN_CORNERS, INDIVIDUAL_EXTENT,           \
                               ISOTROPIC_EXTENT, HAS_IMPORTANCE>(          \
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
