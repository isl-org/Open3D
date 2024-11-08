// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "open3d/ml/paddle/PaddleHelper.h"

using namespace open3d::ml::impl;

namespace {

template <class TReal, class TFeat>
class OutputAllocator {
public:
    OutputAllocator(paddle::Place place) : place(place) {}

    void AllocPooledPositions(TReal** ptr, size_t num) {
        if (num != 0) {
            positions = paddle::empty({int64_t(num), 3}, ToPaddleDtype<TReal>(),
                                      place);
        } else {
            positions = InitializedEmptyTensor(ToPaddleDtype<TReal>(), {0, 3},
                                               place);
        }
        *ptr = positions.data<TReal>();
    }

    void AllocPooledFeatures(TFeat** ptr, size_t num, size_t channels) {
        if (num != 0) {
            features = paddle::empty({int64_t(num), int64_t(channels)},
                                     ToPaddleDtype<TFeat>(), place);
        } else {
            features = InitializedEmptyTensor(ToPaddleDtype<TFeat>(),
                                              {0, int64_t(channels)}, place);
        }
        *ptr = features.data<TFeat>();
    }

    const paddle::Tensor& PooledPositions() const { return positions; }
    const paddle::Tensor& PooledFeatures() const { return features; }

private:
    paddle::Tensor positions;
    paddle::Tensor features;
    paddle::Place place;
};

}  // namespace

template <class TReal, class TFeat>
std::vector<paddle::Tensor> VoxelPoolingCPU(const paddle::Tensor& positions,
                                            const paddle::Tensor& features,
                                            const double voxel_size,
                                            const AccumulationFn position_fn,
                                            const AccumulationFn feature_fn,
                                            const bool debug) {
    OutputAllocator<TReal, TFeat> output_allocator(positions.place());

    if (debug) {
        std::string err;
        PD_CHECK(CheckVoxelSize(err, positions.shape()[0],
                                positions.data<TReal>(), TReal(voxel_size)),
                 err);
    }

    VoxelPooling<TReal, TFeat>(positions.shape()[0], positions.data<TReal>(),
                               features.shape()[1], features.data<TFeat>(),
                               voxel_size, output_allocator, position_fn,
                               feature_fn);

    return {output_allocator.PooledPositions(),
            output_allocator.PooledFeatures()};
}
#define INSTANTIATE(TReal, TFeat)                                       \
    template std::vector<paddle::Tensor> VoxelPoolingCPU<TReal, TFeat>( \
            const paddle::Tensor&, const paddle::Tensor&, const double, \
            const AccumulationFn, const AccumulationFn, const bool);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)
INSTANTIATE(float, float)
INSTANTIATE(float, double)
INSTANTIATE(double, int32_t)
INSTANTIATE(double, int64_t)
INSTANTIATE(double, float)
INSTANTIATE(double, double)
#undef INSTANTIATE

template <class TReal, class TFeat>
void VoxelPoolingGradCPU(paddle::Tensor& features_backprop,
                         const paddle::Tensor& positions,
                         const paddle::Tensor& features,
                         const paddle::Tensor& pooled_positions,
                         const paddle::Tensor& pooled_features_gradient,
                         const double voxel_size,
                         const AccumulationFn position_fn,
                         const AccumulationFn feature_fn) {
    VoxelPoolingBackprop<TReal, TFeat>(
            features_backprop.data<TFeat>(), positions.shape()[0],
            positions.data<TReal>(), features.shape()[1],
            features.data<TFeat>(), pooled_positions.shape()[0],
            pooled_positions.data<TReal>(),
            pooled_features_gradient.data<TFeat>(), TReal(voxel_size),
            position_fn, feature_fn);
}
#define INSTANTIATE(TReal, TFeat)                                          \
    template void VoxelPoolingGradCPU<TReal, TFeat>(                       \
            paddle::Tensor&, const paddle::Tensor&, const paddle::Tensor&, \
            const paddle::Tensor&, const paddle::Tensor&, const double,    \
            const AccumulationFn, const AccumulationFn);
INSTANTIATE(float, float)
INSTANTIATE(float, double)
INSTANTIATE(double, float)
INSTANTIATE(double, double)
