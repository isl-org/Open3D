// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/Macro.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

/// Assert Tensor's dtype is the same as the expected dtype. When an error
/// occurs, the corresponding file name, line number and function name will be
/// printed in the error message.
///
/// Example: check that the tensor has dtype Float32
/// core::AssertTensorDtype(tensor, core::Float32);
#define AssertTensorDtype(tensor, ...)                                     \
    tensor_check::AssertTensorDtype_(                                      \
            __FILE__, __LINE__, static_cast<const char*>(OPEN3D_FUNCTION), \
            tensor, __VA_ARGS__)

/// Assert Tensor's dtype is among one of the expected dtypes. When an error
/// occurs, the corresponding file name, line number and function name will be
/// printed in the error message.
///
/// Example: check that the tensor has dtype Float32 or Float64
/// core::AssertTensorDtypes(tensor, {core::Float32, core::Float64});
#define AssertTensorDtypes(tensor, ...)                                    \
    tensor_check::AssertTensorDtypes_(                                     \
            __FILE__, __LINE__, static_cast<const char*>(OPEN3D_FUNCTION), \
            tensor, __VA_ARGS__)

/// Assert Tensor's device is the same as the expected device. When an error
/// occurs, the corresponding file name, line number and function name will be
/// printed in the error message.
///
/// Example: check that the tensor has device CUDA:0
/// core::AssertTensorDevice(tensor, core::Device("CUDA:0"));
#define AssertTensorDevice(tensor, ...)                                    \
    tensor_check::AssertTensorDevice_(                                     \
            __FILE__, __LINE__, static_cast<const char*>(OPEN3D_FUNCTION), \
            tensor, __VA_ARGS__)

/// Assert Tensor's shape is the same as the expected shape. AssertTensorShape
/// takes a shape (SizeVector) or dynamic shape (DynamicSizeVector). When an
/// error occurs, the corresponding file name, line number and function name
/// will be printed in the error message.
///
/// Example: check that the tensor has shape {100, 3}
/// core::AssertTensorShape(tensor, {100, 3});
///
/// Example: check that the tensor has shape {N, 3}
/// core::AssertTensorShape(tensor, {utility::nullopt, 3});
#define AssertTensorShape(tensor, ...)                                     \
    tensor_check::AssertTensorShape_(                                      \
            __FILE__, __LINE__, static_cast<const char*>(OPEN3D_FUNCTION), \
            tensor, __VA_ARGS__)

namespace open3d {
namespace core {
namespace tensor_check {

void AssertTensorDtype_(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const Dtype& dtype);

void AssertTensorDtypes_(const char* file,
                         int line,
                         const char* function,
                         const Tensor& tensor,
                         const std::vector<Dtype>& dtypes);

void AssertTensorDevice_(const char* file,
                         int line,
                         const char* function,
                         const Tensor& tensor,
                         const Device& device);

void AssertTensorShape_(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const DynamicSizeVector& shape);

}  // namespace tensor_check
}  // namespace core
}  // namespace open3d
