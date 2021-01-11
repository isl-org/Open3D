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

#include "open3d/t/pipelines/kernel/FillInLinearSystem.h"

namespace open3d {
namespace t {
namespace pipelines {
namespace kernel {

void FillInRigidAlignmentTerm(core::Tensor &AtA,
                              core::Tensor &Atb,
                              const core::Tensor &points_i,
                              const core::Tensor &points_j,
                              const core::Tensor &normals_i,
                              int i,
                              int j) {
    core::Device device = AtA.GetDevice();
    if (Atb.GetDevice() != device) {
        utility::LogError("AtA should have the same device as Atb");
    }
    if (points_i.GetDevice() != device) {
        utility::LogError(
                "Points_i should have the same device as the linear system");
    }
    if (points_j.GetDevice() != device) {
        utility::LogError(
                "Points_j should have the same device as the linear system");
    }
    if (normals_i.GetDevice() != device) {
        utility::LogError(
                "Normals_i should have the same device as the linear system");
    }

    core::Device::DeviceType device = device.GetType();
    if (device_type == core::Device::DeviceType::CPU) {
        FillInRigidAlignmentTermCPU(AtA, Atb, points_i, points_j, normals_i, i,
                                    j);

    } else if (device_type == core::Device::DeviceType::CUDA) {
#ifdef BUILD_CUDA_MODULE
        FillInRigidAlignmentTermCUDA(AtA, Atb, points_i, points_j, normals_i, i,
                                     j);

#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("Unimplemented device");
    }
}

}  // namespace kernel
}  // namespace pipelines
}  // namespace t
}  // namespace open3d
