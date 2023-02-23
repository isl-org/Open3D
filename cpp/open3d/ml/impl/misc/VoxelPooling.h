// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <tbb/task_group.h>

#include <Eigen/Core>
#include <unordered_map>

#include "open3d/utility/Helper.h"

namespace open3d {
namespace ml {
namespace impl {

enum AccumulationFn { AVERAGE = 0, NEAREST_NEIGHBOR, MAX, CENTER };

template <class TReal,
          class TFeat,
          AccumulationFn POS_FN,
          AccumulationFn FEAT_FN>
class Accumulator {
public:
    Accumulator()
        : count_(0),
          min_sqr_dist_to_center_(std::numeric_limits<TReal>::max()),
          position_(0, 0, 0) {
        static_assert(POS_FN != MAX, "MAX is not allowed for point positions");
        static_assert(FEAT_FN != CENTER,
                      "CENTER is not allowed for feature vectors");
    }

    template <class Derived, class Derived2, class Derived3>
    inline void AddPoint(const Eigen::MatrixBase<Derived>& pos,
                         const Eigen::MatrixBase<Derived2>& voxel_center,
                         const Eigen::ArrayBase<Derived3>& feat) {
        bool new_nearest_neighbor = false;
        TReal sqr_d = 0;
        if (POS_FN == NEAREST_NEIGHBOR || FEAT_FN == NEAREST_NEIGHBOR) {
            sqr_d = (voxel_center - pos).squaredNorm();
            if (sqr_d < min_sqr_dist_to_center_) {
                new_nearest_neighbor = true;
                min_sqr_dist_to_center_ = sqr_d;
            }
        }
        if (POS_FN == AVERAGE) {
            position_ += pos.array();
        } else if (POS_FN == NEAREST_NEIGHBOR && new_nearest_neighbor) {
            position_ = pos;
        } else if (POS_FN == CENTER) {
            if (count_ == 0) position_ = voxel_center;
        }

        if (count_ == 0) {
            features_.resizeLike(feat);
            features_.setZero();
        }
        if (FEAT_FN == AVERAGE) {
            features_ += feat;
        } else if (FEAT_FN == NEAREST_NEIGHBOR && new_nearest_neighbor) {
            features_ = feat;
        } else if (FEAT_FN == MAX) {
            features_ = features_.max(feat);
        }
        ++count_;
    }

    inline Eigen::Array<TReal, 3, 1> Position() const {
        if (POS_FN == AVERAGE) {
            return position_ / count_;
        } else  // if( POS_FN == NEAREST_NEIGHBOR || POS_FN == CENTER )
        {
            return position_;
        }
    }

    inline Eigen::Array<TFeat, Eigen::Dynamic, 1> Features() const {
        if (FEAT_FN == AVERAGE) {
            return features_ / count_;
        } else  // if( FEAT_FN == NEAREST_NEIGHBOR || FEAT_FN == MAX )
        {
            return features_;
        }
    }

    inline int Count() const { return count_; }

private:
    int count_;
    TReal min_sqr_dist_to_center_;
    Eigen::Array<TReal, 3, 1> position_;
    Eigen::Array<TFeat, Eigen::Dynamic, 1> features_;
};  // namespace

template <class TReal,
          class TFeat,
          AccumulationFn POS_FN,
          AccumulationFn FEAT_FN>
class AccumulatorBackprop {
public:
    AccumulatorBackprop()
        : count_(0),
          min_sqr_dist_to_center_(std::numeric_limits<TReal>::max()),
          position_(0, 0, 0) {
        static_assert(POS_FN != MAX, "MAX is not allowed for point positions");
        static_assert(FEAT_FN != CENTER,
                      "CENTER is not allowed for feature vectors");
    }

    template <class Derived, class Derived2, class Derived3>
    inline void AddPoint(const Eigen::MatrixBase<Derived>& pos,
                         const Eigen::MatrixBase<Derived2>& voxel_center,
                         const Eigen::ArrayBase<Derived3>& feat,
                         const size_t idx) {
        bool new_nearest_neighbor = false;
        TReal sqr_d = 0;
        if (POS_FN == NEAREST_NEIGHBOR || FEAT_FN == NEAREST_NEIGHBOR) {
            sqr_d = (voxel_center - pos).squaredNorm();
            if (sqr_d < min_sqr_dist_to_center_) {
                new_nearest_neighbor = true;
                min_sqr_dist_to_center_ = sqr_d;
            }
        }

        if (POS_FN == AVERAGE) {
            position_ += pos.array();
        } else if (POS_FN == NEAREST_NEIGHBOR && new_nearest_neighbor) {
            position_ = pos;
        } else if (POS_FN == CENTER) {
            if (count_ == 0) position_ = voxel_center;
        }

        if (count_ == 0) {
            features_.resizeLike(feat);
            features_.setZero();
            if (FEAT_FN == NEAREST_NEIGHBOR) {
                features_ = feat;
                index_.resize(1);
                index_(0) = idx;
                ++count_;
                return;
            } else if (FEAT_FN == MAX) {
                features_ = feat;
                index_.resizeLike(feat);
                index_ = idx;
                ++count_;
                return;
            }
        }
        if (FEAT_FN == AVERAGE) {
            features_ += feat;
        } else if (FEAT_FN == NEAREST_NEIGHBOR && new_nearest_neighbor) {
            features_ = feat;
            index_(0) = idx;
        } else if (FEAT_FN == MAX) {
            for (int i = 0; i < features_.rows(); ++i) {
                if (feat(i) > features_(i)) {
                    features_(i) = feat(i);
                    index_(i) = idx;
                }
            }
        }
        ++count_;
    }

    inline Eigen::Array<TReal, 3, 1> Position() const {
        if (POS_FN == AVERAGE) {
            return position_ / count_;
        } else  // if( POS_FN == NEAREST_NEIGHBOR || POS_FN == CENTER )
        {
            return position_;
        }
    }

    inline Eigen::Array<TFeat, Eigen::Dynamic, 1> Features() const {
        if (FEAT_FN == AVERAGE) {
            return features_ / count_;
        } else  // if( FEAT_FN == NEAREST_NEIGHBOR || FEAT_FN == MAX )
        {
            return features_;
        }
    }

    inline int Count() const { return count_; }

    inline Eigen::Array<size_t, Eigen::Dynamic, 1> Index() const {
        return index_;
    }

private:
    int count_;
    TReal min_sqr_dist_to_center_;
    Eigen::Array<TReal, 3, 1> position_;
    Eigen::Array<TFeat, Eigen::Dynamic, 1> features_;
    Eigen::Array<size_t, Eigen::Dynamic, 1> index_;
};

/// Function for debugging. Checks if the voxel size is too small.
template <class T>
bool CheckVoxelSize(std::string& err,
                    const size_t num_positions,
                    const T* const positions,
                    const T voxel_size) {
    typedef Eigen::Array<double, 3, 1> Vec3_t;
    if (num_positions == 0) {
        return true;
    }

    Vec3_t bb_min, bb_max;
    bb_min << positions[0], positions[1], positions[2];
    bb_max = bb_min;

    Vec3_t voxel_size3(voxel_size, voxel_size, voxel_size);

    for (size_t i = 1; i < num_positions; ++i) {
        Vec3_t pos(positions[i * 3 + 0], positions[i * 3 + 1],
                   positions[i * 3 + 2]);
        bb_min = bb_min.min(pos);
        bb_max = bb_max.max(pos);
    }

    // the min and max bounding box shall be a multiple of the voxel size
    bb_min /= voxel_size3;
    bb_min = bb_min.floor() * voxel_size3;
    bb_max /= voxel_size3;
    bb_max = bb_max.ceil() * voxel_size3;

    if (voxel_size * double(std::numeric_limits<int>::max()) <
                bb_max.maxCoeff() ||
        voxel_size * double(std::numeric_limits<int>::min()) >
                bb_min.maxCoeff()) {
        err = "voxel_size is too small\n";
        return false;
    }
    return true;
}

/// Computes an integer voxel index for a 3D position.
///
/// \param pos               A 3D position.
/// \param inv_voxel_size    The reciprocal of the voxel size
///
template <class TDerived>
Eigen::Vector3i ComputeVoxelIndex(
        const Eigen::ArrayBase<TDerived>& pos,
        const typename TDerived::Scalar& inv_voxel_size) {
    typedef typename TDerived::Scalar Scalar_t;
    Eigen::Array<Scalar_t, 3, 1> ref_coord = pos * inv_voxel_size;

    Eigen::Vector3i voxel_index;
    voxel_index = ref_coord.floor().template cast<int>();
    return voxel_index;
}

// implementation for VoxelPooling with template parameter for the accumulator.
template <class TReal, class TFeat, class ACCUMULATOR, class OUTPUT_ALLOCATOR>
void _VoxelPooling(size_t num_inp,
                   const TReal* const inp_positions,
                   int in_channels,
                   const TFeat* inp_features,
                   TReal voxel_size,
                   OUTPUT_ALLOCATOR& output_allocator) {
    if (num_inp == 0) {
        TReal* out_pos_ptr;
        TFeat* out_feat_ptr;
        output_allocator.AllocPooledPositions(&out_pos_ptr, 0);
        output_allocator.AllocPooledFeatures(&out_feat_ptr, 0, in_channels);
        return;
    }

    typedef Eigen::Array<TReal, 3, 1> Vec3_t;
    typedef Eigen::Array<TFeat, Eigen::Dynamic, 1> FeatureVec_t;

    std::unordered_map<Eigen::Vector3i, ACCUMULATOR,
                       open3d::utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;

    Vec3_t voxel_center;
    Eigen::Vector3i voxel_index;
    TReal inv_voxel_size = 1 / voxel_size;
    TReal half_voxel_size = 0.5 * voxel_size;
    for (size_t i = 0; i < num_inp; ++i) {
        Eigen::Map<const Vec3_t> pos(inp_positions + i * 3);

        voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);

        voxel_center << voxel_index(0) * voxel_size + half_voxel_size,
                voxel_index(1) * voxel_size + half_voxel_size,
                voxel_index(2) * voxel_size + half_voxel_size;

        Eigen::Map<const FeatureVec_t> feat(inp_features + in_channels * i,
                                            in_channels);
        voxelindex_to_accpoint[voxel_index].AddPoint(
                pos.matrix(), voxel_center.matrix(), feat);
    }

    const size_t num_out = voxelindex_to_accpoint.size();

    TReal* out_pos_ptr;
    TFeat* out_feat_ptr;
    output_allocator.AllocPooledPositions(&out_pos_ptr, num_out);
    output_allocator.AllocPooledFeatures(&out_feat_ptr, num_out, in_channels);

    size_t i = 0;
    for (const auto point : voxelindex_to_accpoint) {
        Vec3_t pos = point.second.Position();
        Eigen::Map<Vec3_t> out_pos(out_pos_ptr + i * 3);
        out_pos = pos;

        Eigen::Map<FeatureVec_t> out_feat(out_feat_ptr + i * in_channels,
                                          in_channels);
        out_feat = point.second.Features();
        ++i;
    }
}

// implementation for VoxelPoolingBackprop with template parameter for the
// accumulator.
template <class TReal, class TFeat, class ACCUMULATOR, AccumulationFn FEAT_FN>
void _VoxelPoolingBackprop(TFeat* features_backprop,
                           size_t num_inp,
                           const TReal* const inp_positions,
                           int in_channels,
                           const TFeat* const inp_features,
                           size_t num_pooled,
                           const TReal* const pooled_positions,
                           const TFeat* const pooled_features_gradient,
                           TReal voxel_size) {
    if (num_inp == 0) {
        return;
    }
    memset(features_backprop, 0, sizeof(TFeat) * num_inp * in_channels);

    typedef Eigen::Array<TReal, 3, 1> Vec3_t;
    typedef Eigen::Array<TFeat, Eigen::Dynamic, 1> FeatureVec_t;

    Vec3_t voxel_size3(voxel_size, voxel_size, voxel_size);

    // create the two hash maps in parallel
    tbb::task_group task_group;

    std::unordered_map<Eigen::Vector3i, ACCUMULATOR,
                       open3d::utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;

    task_group.run([&] {
        Vec3_t voxel_center;
        Eigen::Vector3i voxel_index;
        TReal inv_voxel_size = 1 / voxel_size;
        TReal half_voxel_size = 0.5 * voxel_size;
        for (size_t i = 0; i < num_inp; ++i) {
            Eigen::Map<const Vec3_t> pos(inp_positions + i * 3);

            voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);

            voxel_center << voxel_index(0) * voxel_size + half_voxel_size,
                    voxel_index(1) * voxel_size + half_voxel_size,
                    voxel_index(2) * voxel_size + half_voxel_size;

            Eigen::Map<const FeatureVec_t> feat(inp_features + in_channels * i,
                                                in_channels);
            voxelindex_to_accpoint[voxel_index].AddPoint(
                    pos.matrix(), voxel_center.matrix(), feat, i);
        }
    });

    std::unordered_map<Eigen::Vector3i, size_t,
                       open3d::utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_gradindex;

    task_group.run([&] {
        Eigen::Vector3i voxel_index;
        TReal inv_voxel_size = 1 / voxel_size;
        for (size_t i = 0; i < num_pooled; ++i) {
            Eigen::Map<const Vec3_t> pos(pooled_positions + i * 3);

            voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);

            voxelindex_to_gradindex[voxel_index] = i;
        }
    });

    task_group.wait();

    if (FEAT_FN == AVERAGE) {
        Eigen::Vector3i voxel_index;
        TReal inv_voxel_size = 1 / voxel_size;
        for (size_t i = 0; i < num_inp; ++i) {
            Eigen::Map<const Vec3_t> pos(inp_positions + i * 3);

            voxel_index = ComputeVoxelIndex(pos, inv_voxel_size);

            Eigen::Map<FeatureVec_t> feat_bp(
                    features_backprop + in_channels * i, in_channels);

            size_t grad_idx = voxelindex_to_gradindex[voxel_index];
            int count = voxelindex_to_accpoint[voxel_index].Count();
            Eigen::Map<const FeatureVec_t> grad(
                    pooled_features_gradient + in_channels * grad_idx,
                    in_channels);
            feat_bp = grad / count;
        }
    }

    if (FEAT_FN == NEAREST_NEIGHBOR) {
        for (const auto point : voxelindex_to_accpoint) {
            size_t idx = point.second.Index()(0);
            Eigen::Map<FeatureVec_t> feat_bp(
                    features_backprop + in_channels * idx, in_channels);

            size_t grad_idx = voxelindex_to_gradindex[point.first];
            Eigen::Map<const FeatureVec_t> grad(
                    pooled_features_gradient + in_channels * grad_idx,
                    in_channels);
            feat_bp = grad;
        }
    }

    if (FEAT_FN == MAX) {
        for (const auto point : voxelindex_to_accpoint) {
            size_t grad_idx = voxelindex_to_gradindex[point.first];
            Eigen::Map<const FeatureVec_t> grad(
                    pooled_features_gradient + in_channels * grad_idx,
                    in_channels);
            for (int i = 0; i < in_channels; ++i) {
                size_t idx = point.second.Index()(i);
                Eigen::Map<FeatureVec_t> feat_bp(
                        features_backprop + in_channels * idx, in_channels);
                feat_bp(i) = grad(i);
            }
        }
    }
}

/// Pooling operation for point clouds. Aggregates points that are inside the
/// same voxel.
///
/// \tparam TReal    Scalar type for point positions.
///
/// \tparam TFeat    Scalar type for the features.
///
/// \tparam OUTPUT_ALLOCATOR    Type of the output_allocator. See
///         \p output_allocator for more information.
///
/// \param num_inp    The number of input points.
///
/// \param inp_positions    Array with 3D point positions.
///
/// \param in_channels    The number of input feature channels.
///
/// \param inp_features    The array with the point features. The shape is
///                        [num_inp, in_channels].
///
/// \param voxel_size    The voxel size (voxel edge length) used for pooling.
///
/// \param output_allocator    An object that implements functions for
///         allocating the output arrays. The object must implement functions
///         AllocPooledPositions(TReal** ptr, size_t num) and
///         AllocPooledFeatures(TFeat** ptr, size_t num, channels), with 'num'
///         as the number of output points and 'channels' as \p in_channels .
///         Both functions should allocate memory and return a pointer
///         to that memory in ptr.
///         Both functions must accept the argument num==0. In this case ptr
///         does not need to be set. Note that AllocPooledPositions must
///         allocate memory for num*3*sizeof(TReal) bytes.
///
/// \param position_fn    Defines how the new point positions will be computed.
///        AVERAGE computes the center of gravity for the points within one
///        voxel.
///        NEAREST_NEIGHBOR selects the point closest to the voxel center.
///        CENTER uses the voxel center for the position of the generated point.
///
/// \param feature_fn    Defines how the new features will be computed.
///        AVERAGE computes the average feature vector.
///        NEAREST_NEIGHBOR selects the feature vector of the point closest to
///        the voxel center.
///        MAX uses the maximum feature among all points within the voxel.
///
template <class TReal, class TFeat, class OUTPUT_ALLOCATOR>
void VoxelPooling(size_t num_inp,
                  const TReal* const inp_positions,
                  int in_channels,
                  const TFeat* inp_features,
                  TReal voxel_size,
                  OUTPUT_ALLOCATOR& output_allocator,
                  AccumulationFn position_fn,
                  AccumulationFn feature_fn) {
#define CALL_TEMPLATE(POS_FN, FEAT_FN)                                         \
    if (POS_FN == position_fn && FEAT_FN == feature_fn) {                      \
        _VoxelPooling<TReal, TFeat,                                            \
                      Accumulator<TReal, TFeat, POS_FN, FEAT_FN>>(             \
                num_inp, inp_positions, in_channels, inp_features, voxel_size, \
                output_allocator);                                             \
    }

    CALL_TEMPLATE(AVERAGE, AVERAGE)
    CALL_TEMPLATE(AVERAGE, NEAREST_NEIGHBOR)
    CALL_TEMPLATE(AVERAGE, MAX)
    CALL_TEMPLATE(NEAREST_NEIGHBOR, AVERAGE)
    CALL_TEMPLATE(NEAREST_NEIGHBOR, NEAREST_NEIGHBOR)
    CALL_TEMPLATE(NEAREST_NEIGHBOR, MAX)
    CALL_TEMPLATE(CENTER, AVERAGE)
    CALL_TEMPLATE(CENTER, NEAREST_NEIGHBOR)
    CALL_TEMPLATE(CENTER, MAX)

#undef CALL_TEMPLATE
}

/// Backpropagation to the features for VoxelPooling.
///
/// \param features_backprop    The output array with the gradients for the
///        features.
///
/// \param num_pooled    The number of points after pooling with \p
///         VoxelPooling.
///
/// \param pooled_positions    Array with the 3D positions after pooling.
///
/// \param pooled_features_gradient    Array with the point features after
///        pooling.
///
/// See \p VoxelPooling for the description of the remaining parameters.
///
template <class TReal, class TFeat>
void VoxelPoolingBackprop(TFeat* features_backprop,
                          size_t num_inp,
                          const TReal* const inp_positions,
                          int in_channels,
                          const TFeat* const inp_features,
                          size_t num_pooled,
                          const TReal* const pooled_positions,
                          const TFeat* const pooled_features_gradient,
                          TReal voxel_size,
                          AccumulationFn position_fn,
                          AccumulationFn feature_fn) {
#define CALL_TEMPLATE(POS_FN, FEAT_FN)                                        \
    if (POS_FN == position_fn && FEAT_FN == feature_fn) {                     \
        _VoxelPoolingBackprop<                                                \
                TReal, TFeat,                                                 \
                AccumulatorBackprop<TReal, TFeat, POS_FN, FEAT_FN>, FEAT_FN>( \
                features_backprop, num_inp, inp_positions, in_channels,       \
                inp_features, num_pooled, pooled_positions,                   \
                pooled_features_gradient, voxel_size);                        \
    }

    CALL_TEMPLATE(AVERAGE, AVERAGE)
    CALL_TEMPLATE(AVERAGE, NEAREST_NEIGHBOR)
    CALL_TEMPLATE(AVERAGE, MAX)
    CALL_TEMPLATE(NEAREST_NEIGHBOR, AVERAGE)
    CALL_TEMPLATE(NEAREST_NEIGHBOR, NEAREST_NEIGHBOR)
    CALL_TEMPLATE(NEAREST_NEIGHBOR, MAX)
    CALL_TEMPLATE(CENTER, AVERAGE)
    CALL_TEMPLATE(CENTER, NEAREST_NEIGHBOR)
    CALL_TEMPLATE(CENTER, MAX)

#undef CALL_TEMPLATE
}

}  // namespace impl
}  // namespace ml
}  // namespace open3d
