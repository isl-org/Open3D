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
//

#include "open3d/ml/impl/misc/VoxelPooling.h"
#include "open3d/ml/pytorch/TorchHelper.h"
#include "torch/script.h"

using namespace open3d::ml::impl;

namespace {

template <class TReal, class TFeat>
class OutputAllocator {
public:
    OutputAllocator(torch::DeviceType device_type, int device_idx)
        : device_type(device_type), device_idx(device_idx) {}

    void AllocPooledPositions(TReal** ptr, size_t num) {
        positions = torch::empty({int64_t(num), 3},
                                 torch::dtype(ToTorchDtype<TReal>())
                                         .device(device_type, device_idx));
        *ptr = positions.data_ptr<TReal>();
    }

    void AllocPooledFeatures(TFeat** ptr, size_t num, size_t channels) {
        features = torch::empty({int64_t(num), int64_t(channels)},
                                torch::dtype(ToTorchDtype<TFeat>())
                                        .device(device_type, device_idx));
        *ptr = features.data_ptr<TFeat>();
    }

    const torch::Tensor& PooledPositions() const { return positions; }
    const torch::Tensor& PooledFeatures() const { return features; }

private:
    torch::Tensor positions;
    torch::Tensor features;
    torch::DeviceType device_type;
    int device_idx;
};

}  // namespace

template <class TReal, class TFeat>
std::tuple<torch::Tensor, torch::Tensor> VoxelPoolingCPU(
        const torch::Tensor& positions,
        const torch::Tensor& features,
        const double voxel_size,
        const AccumulationFn position_fn,
        const AccumulationFn feature_fn,
        const bool debug) {
    OutputAllocator<TReal, TFeat> output_allocator(positions.device().type(),
                                                   positions.device().index());

    if (debug) {
        std::string err;
        TORCH_CHECK(
                CheckVoxelSize(err, positions.size(0),
                               positions.data_ptr<TReal>(), TReal(voxel_size)),
                err);
    }

    VoxelPooling<TReal, TFeat>(positions.size(0), positions.data_ptr<TReal>(),
                               features.size(1), features.data_ptr<TFeat>(),
                               voxel_size, output_allocator, position_fn,
                               feature_fn);

    return std::make_tuple(output_allocator.PooledPositions(),
                           output_allocator.PooledFeatures());
}
#define INSTANTIATE(TReal, TFeat)                                             \
    template std::tuple<torch::Tensor, torch::Tensor>                         \
    VoxelPoolingCPU<TReal, TFeat>(const torch::Tensor&, const torch::Tensor&, \
                                  const double, const AccumulationFn,         \
                                  const AccumulationFn, const bool);

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
void VoxelPoolingGradCPU(torch::Tensor& features_backprop,
                         const torch::Tensor& positions,
                         const torch::Tensor& features,
                         const torch::Tensor& pooled_positions,
                         const torch::Tensor& pooled_features_gradient,
                         const double voxel_size,
                         const AccumulationFn position_fn,
                         const AccumulationFn feature_fn) {
    VoxelPoolingBackprop<TReal, TFeat>(
            features_backprop.data_ptr<TFeat>(), positions.size(0),
            positions.data_ptr<TReal>(), features.size(1),
            features.data_ptr<TFeat>(), pooled_positions.size(0),
            pooled_positions.data_ptr<TReal>(),
            pooled_features_gradient.data_ptr<TFeat>(), TReal(voxel_size),
            position_fn, feature_fn);
}
#define INSTANTIATE(TReal, TFeat)                                       \
    template void VoxelPoolingGradCPU<TReal, TFeat>(                    \
            torch::Tensor&, const torch::Tensor&, const torch::Tensor&, \
            const torch::Tensor&, const torch::Tensor&, const double,   \
            const AccumulationFn, const AccumulationFn);
INSTANTIATE(float, float)
INSTANTIATE(float, double)
INSTANTIATE(double, float)
INSTANTIATE(double, double)
