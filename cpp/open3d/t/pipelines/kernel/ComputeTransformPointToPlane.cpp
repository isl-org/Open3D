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
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/t/pipelines/kernel/ComputeTransformPointToPlane.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

core::Tensor ComputeTransformPointToPlane(
        const core::Tensor &source_points_alligned,
        const core::Tensor &target_points_alligned,
        const core::Tensor &target_normals_alligned,
        const core::Dtype dtype,
        const core::Device device) {
    // transformation tensor [ouput]
    core::Tensor transformation = core::Tensor::Zeros({4, 4}, dtype, device);

    // number of correspondences
    int n = source_points_alligned.GetShape()[0];

    // pointer to point cloud data - alligned according to correspondences
    const float *src_pcd_ptr = static_cast<const float *>(
            source_points_alligned.Contiguous().GetDataPtr());
    const float *tar_pcd_ptr = static_cast<const float *>(
            target_points_alligned.Contiguous().GetDataPtr());
    const float *tar_norm_ptr = static_cast<const float *>(
            target_normals_alligned.Contiguous().GetDataPtr());

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ComputeTransformPointToPlaneCPU(src_pcd_ptr, tar_pcd_ptr, tar_norm_ptr,
                                        n, transformation, dtype, device);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ComputeTransformPointToPlaneCUDA(src_pcd_ptr, tar_pcd_ptr, tar_norm_ptr,
                                         n, transformation, dtype, device);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
    return transformation;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
