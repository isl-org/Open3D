// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EposePRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/TransformationConverter.h"

#include <cmath>

#include "open3d/core/Tensor.h"
#include "open3d/t/pipelines/kernel/TransformationConverterImpl.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

core::Tensor RtToTransformation(const core::Tensor &R, const core::Tensor &t) {
    core::Dtype dtype = core::Dtype::Float32;
    core::Device device = R.GetDevice();
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    R.AssertShape({3, 3});
    R.AssertDtype(dtype);
    t.AssertShape({3});
    t.AssertDevice(device);
    t.AssertDtype(dtype);

    // Rotation.
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(0, 3, 1)},
            R);
    // Translation.
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            t.Reshape({3, 1}));
    // Scale [assumed to be 1].
    transformation[3][3] = 1;
    return transformation;
}

core::Tensor PoseToTransformation(const core::Tensor &pose) {
    core::Dtype dtype = core::Dtype::Float32;
    pose.AssertShape({6});
    pose.AssertDtype(dtype);
    core::Device device = pose.GetDevice();
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);
    transformation = transformation.Contiguous();
    core::Tensor pose_ = pose.Contiguous();
    float *transformation_ptr =
            static_cast<float *>(transformation.GetDataPtr());
    const float *pose_ptr = static_cast<const float *>(pose_.GetDataPtr());

    // Rotation from pose.
    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        PoseToTransformationImpl(transformation_ptr, pose_ptr);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        PoseToTransformationCUDA(transformation_ptr, pose_ptr);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
    // Translation from pose.
    transformation.SetItem(
            {core::TensorKey::Slice(0, 3, 1), core::TensorKey::Slice(3, 4, 1)},
            pose_.GetItem({core::TensorKey::Slice(3, 6, 1)}).Reshape({3, 1}));
    // Scale [assumed to be 1].
    transformation[3][3] = 1;
    return transformation;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
