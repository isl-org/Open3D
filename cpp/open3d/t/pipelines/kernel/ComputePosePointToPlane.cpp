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

#include "open3d/t/pipelines/kernel/ComputePosePointToPlane.h"

#include "open3d/t/pipelines/kernel/ComputePosePointToPlaneImp.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

core::Tensor ComputePosePointToPlane(
        const core::Tensor &source_points,
        const core::Tensor &target_points,
        const core::Tensor &target_normals,
        const pipelines::registration::CorrespondenceSet &corres) {
    // Get dtype and device.
    core::Dtype dtype = source_points.GetDtype();
    core::Device device = source_points.GetDevice();

    // Checks.
    // TODO: These checks are redundant, so provide a environment
    // variable based method to skip these redundant tests.
    target_points.AssertDtype(dtype);
    target_normals.AssertDtype(dtype);
    target_points.AssertDevice(device);
    target_normals.AssertDevice(device);

    // Pose {6,} tensor [ouput].
    core::Tensor pose = core::Tensor::Empty({6}, dtype, device);
    // Number of correspondences.
    int n = corres.first.GetShape()[0];

    // Pointer to point cloud data - indexed according to correspondences.
    // Pointer to point cloud data - indexed according to correspondences.
    core::Tensor source_points_contiguous = source_points.Contiguous();
    core::Tensor target_points_contiguous = target_points.Contiguous();
    core::Tensor target_normals_contiguous = target_normals.Contiguous();
    core::Tensor corres_first_contiguous = corres.first.Contiguous();
    core::Tensor corres_second_contiguous = corres.second.Contiguous();

    const float *src_pcd_ptr = source_points_contiguous.GetDataPtr<float>();
    const float *tar_pcd_ptr = target_points_contiguous.GetDataPtr<float>();
    const float *tar_norm_ptr = target_normals_contiguous.GetDataPtr<float>();
    const int64_t *corres_first = corres_first_contiguous.GetDataPtr<int64_t>();
    const int64_t *corres_second =
            corres_second_contiguous.GetDataPtr<int64_t>();

    core::Device::DeviceType device_type = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        ComputePosePointToPlaneCPU(src_pcd_ptr, tar_pcd_ptr, tar_norm_ptr,
                                   corres_first, corres_second, n, pose, dtype,
                                   device);
    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        ComputePosePointToPlaneCUDA(src_pcd_ptr, tar_pcd_ptr, tar_norm_ptr,
                                    corres_first, corres_second, n, pose, dtype,
                                    device);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device.");
    }
    return pose;
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
