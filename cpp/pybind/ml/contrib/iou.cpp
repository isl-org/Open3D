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

#include "open3d/ml/contrib/IoU.h"

#include "open3d/utility/Console.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/docstring.h"
#include "pybind/ml/contrib/contrib.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace ml {
namespace contrib {

py::array IouBevCPU(py::array boxes_a, py::array boxes_b) {
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous();
    boxes_a_tensor.AssertDtype(core::Dtype::Float32);
    boxes_a_tensor.AssertShapeCompatible({utility::nullopt, 5});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous();
    boxes_b_tensor.AssertDtype(core::Dtype::Float32);
    boxes_b_tensor.AssertShapeCompatible({utility::nullopt, 5});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Dtype::Float32, core::Device("CPU:0"));

    IoUBevCPUKernel(static_cast<const float*>(boxes_a_tensor.GetDataPtr()),
                    static_cast<const float*>(boxes_b_tensor.GetDataPtr()),
                    static_cast<float*>(iou_tensor.GetDataPtr()), num_a, num_b);

    return core::TensorToPyArray(iou_tensor);
}

py::array Iou3dCPU(py::array boxes_a, py::array boxes_b) {
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous();
    boxes_a_tensor.AssertDtype(core::Dtype::Float32);
    boxes_a_tensor.AssertShapeCompatible({utility::nullopt, 7});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous();
    boxes_b_tensor.AssertDtype(core::Dtype::Float32);
    boxes_b_tensor.AssertShapeCompatible({utility::nullopt, 7});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Dtype::Float32, core::Device("CPU:0"));

    IoU3dCPUKernel(static_cast<const float*>(boxes_a_tensor.GetDataPtr()),
                   static_cast<const float*>(boxes_b_tensor.GetDataPtr()),
                   static_cast<float*>(iou_tensor.GetDataPtr()), num_a, num_b);

    return core::TensorToPyArray(iou_tensor);
}

#ifdef BUILD_CUDA_MODULE
py::array IouBevCUDA(py::array boxes_a, py::array boxes_b) {
    core::Device cuda_device("CUDA:0");
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous().To(cuda_device);
    boxes_a_tensor.AssertDtype(core::Dtype::Float32);
    boxes_a_tensor.AssertShapeCompatible({utility::nullopt, 5});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous().To(cuda_device);
    boxes_b_tensor.AssertDtype(core::Dtype::Float32);
    boxes_b_tensor.AssertShapeCompatible({utility::nullopt, 5});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Dtype::Float32, cuda_device);

    IoUBevCUDAKernel(static_cast<const float*>(boxes_a_tensor.GetDataPtr()),
                     static_cast<const float*>(boxes_b_tensor.GetDataPtr()),
                     static_cast<float*>(iou_tensor.GetDataPtr()), num_a,
                     num_b);
    return core::TensorToPyArray(iou_tensor.To(core::Device("CPU:0")));
}

py::array Iou3dCUDA(py::array boxes_a, py::array boxes_b) {
    core::Device cuda_device("CUDA:0");
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous().To(cuda_device);
    boxes_a_tensor.AssertDtype(core::Dtype::Float32);
    boxes_a_tensor.AssertShapeCompatible({utility::nullopt, 7});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous().To(cuda_device);
    boxes_b_tensor.AssertDtype(core::Dtype::Float32);
    boxes_b_tensor.AssertShapeCompatible({utility::nullopt, 7});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Dtype::Float32, cuda_device);

    IoU3dCUDAKernel(static_cast<const float*>(boxes_a_tensor.GetDataPtr()),
                    static_cast<const float*>(boxes_b_tensor.GetDataPtr()),
                    static_cast<float*>(iou_tensor.GetDataPtr()), num_a, num_b);
    return core::TensorToPyArray(iou_tensor.To(core::Device("CPU:0")));
}
#endif

void pybind_contrib_iou(py::module& m_contrib) {
    m_contrib.def("iou_bev_cpu", &IouBevCPU, "boxes_a"_a, "boxes_b"_a);
    m_contrib.def("iou_3d_cpu", &Iou3dCPU, "boxes_a"_a, "boxes_b"_a);

#ifdef BUILD_CUDA_MODULE
    // These CUDA functions still uses numpy arrays as input and output, i.e.
    // data will be copy to and from the CUDA device.
    m_contrib.def("iou_bev_cuda", &IouBevCUDA, "boxes_a"_a, "boxes_b"_a);
    m_contrib.def("iou_3d_cuda", &Iou3dCUDA, "boxes_a"_a, "boxes_b"_a);
#endif
}

}  // namespace contrib
}  // namespace ml
}  // namespace open3d
