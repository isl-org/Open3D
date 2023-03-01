// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/contrib/IoU.h"

#include "open3d/core/TensorCheck.h"
#include "open3d/utility/Logging.h"
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
    core::AssertTensorDtype(boxes_a_tensor, core::Float32);
    core::AssertTensorShape(boxes_a_tensor, {utility::nullopt, 5});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous();
    core::AssertTensorDtype(boxes_b_tensor, core::Float32);
    core::AssertTensorShape(boxes_b_tensor, {utility::nullopt, 5});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Float32, core::Device("CPU:0"));

    IoUBevCPUKernel(boxes_a_tensor.GetDataPtr<float>(),
                    boxes_b_tensor.GetDataPtr<float>(),
                    iou_tensor.GetDataPtr<float>(), num_a, num_b);

    return core::TensorToPyArray(iou_tensor);
}

py::array Iou3dCPU(py::array boxes_a, py::array boxes_b) {
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous();
    core::AssertTensorDtype(boxes_a_tensor, core::Float32);
    core::AssertTensorShape(boxes_a_tensor, {utility::nullopt, 7});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous();
    core::AssertTensorDtype(boxes_b_tensor, core::Float32);
    core::AssertTensorShape(boxes_b_tensor, {utility::nullopt, 7});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Float32, core::Device("CPU:0"));

    IoU3dCPUKernel(boxes_a_tensor.GetDataPtr<float>(),
                   boxes_b_tensor.GetDataPtr<float>(),
                   iou_tensor.GetDataPtr<float>(), num_a, num_b);

    return core::TensorToPyArray(iou_tensor);
}

#ifdef BUILD_CUDA_MODULE
py::array IouBevCUDA(py::array boxes_a, py::array boxes_b) {
    core::Device cuda_device("CUDA:0");
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous().To(cuda_device);
    core::AssertTensorDtype(boxes_a_tensor, core::Float32);
    core::AssertTensorShape(boxes_a_tensor, {utility::nullopt, 5});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous().To(cuda_device);
    core::AssertTensorDtype(boxes_b_tensor, core::Float32);
    core::AssertTensorShape(boxes_b_tensor, {utility::nullopt, 5});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Float32, cuda_device);

    IoUBevCUDAKernel(boxes_a_tensor.GetDataPtr<float>(),
                     boxes_b_tensor.GetDataPtr<float>(),
                     iou_tensor.GetDataPtr<float>(), num_a, num_b);
    return core::TensorToPyArray(iou_tensor.To(core::Device("CPU:0")));
}

py::array Iou3dCUDA(py::array boxes_a, py::array boxes_b) {
    core::Device cuda_device("CUDA:0");
    core::Tensor boxes_a_tensor =
            core::PyArrayToTensor(boxes_a, true).Contiguous().To(cuda_device);
    core::AssertTensorDtype(boxes_a_tensor, core::Float32);
    core::AssertTensorShape(boxes_a_tensor, {utility::nullopt, 7});
    int64_t num_a = boxes_a_tensor.GetLength();

    core::Tensor boxes_b_tensor =
            core::PyArrayToTensor(boxes_b, true).Contiguous().To(cuda_device);
    core::AssertTensorDtype(boxes_b_tensor, core::Float32);
    core::AssertTensorShape(boxes_b_tensor, {utility::nullopt, 7});
    int64_t num_b = boxes_b_tensor.GetLength();

    core::Tensor iou_tensor = core::Tensor(
            {boxes_a_tensor.GetLength(), boxes_b_tensor.GetLength()},
            core::Float32, cuda_device);

    IoU3dCUDAKernel(boxes_a_tensor.GetDataPtr<float>(),
                    boxes_b_tensor.GetDataPtr<float>(),
                    iou_tensor.GetDataPtr<float>(), num_a, num_b);
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
