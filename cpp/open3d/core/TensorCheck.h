// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#pragma once

#include <string>

#include "open3d/Macro.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

#ifdef _WIN32
#define __FN__ __FUNCSIG__
#else
#define __FN__ __PRETTY_FUNCTION__
#endif

/// Assert Tensor's dtype is the same as the expected dtype.
///
/// Example: core::AssertTensorDtype(tensor, core::Float32);
///          core::AssertTensorDtype(tensor, core::Float32, "Error!");
#define AssertTensorDtype(...)                                                \
    OPEN3D_GET_ARG_4(__VA_ARGS__, _AssertTensorDtype3, _AssertTensorDtype2, ) \
    (__VA_ARGS__)

/// Assert Tensor's device is the same as the expected device.
///
/// Example: core::AssertTensorDtype(tensor, core::Device("CUDA:0"));
///          core::AssertTensorDtype(tensor, core::Device("CUDA:0"), "Error!");
#define AssertTensorDevice(...)                         \
    OPEN3D_GET_ARG_4(__VA_ARGS__, _AssertTensorDevice3, \
                     _AssertTensorDevice2, )            \
    (__VA_ARGS__)

/// Assert Tensor's shape is the same as the expected shape.
///
/// Example: core::AssertTensorDtype(tensor, {2, 3});
///          core::AssertTensorDtype(tensor, {2, 3}, "Error!");
#define AssertTensorShape(...)                                                \
    OPEN3D_GET_ARG_4(__VA_ARGS__, _AssertTensorShape3, _AssertTensorShape2, ) \
    (__VA_ARGS__)

/// Assert Tensor's shape is compatbile with the expected dynamic shape.
///
/// Example: core::AssertTensorShapeCompatible(tensor,
///                                            {2, utility::nullopt});
///          core::AssertTensorShapeCompatible(tensor,
///                                            {2, utility::nullopt}, "Error!");
#define AssertTensorShapeCompatible(...)                         \
    OPEN3D_GET_ARG_4(__VA_ARGS__, _AssertTensorShapeCompatible3, \
                     _AssertTensorShapeCompatible2, )            \
    (__VA_ARGS__)

// Helper functions.
#define _AssertTensorDtype2(tensor, dtype)                                    \
    tensor_check::_AssertTensorDtype(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, dtype)
#define _AssertTensorDtype3(tensor, dtype, message)                           \
    tensor_check::_AssertTensorDtype(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, dtype, message)
#define _AssertTensorDevice2(tensor, device)                                   \
    tensor_check::_AssertTensorDevice(__FILE__, __LINE__, (const char*)__FN__, \
                                      tensor, device)
#define _AssertTensorDevice3(tensor, device, message)                          \
    tensor_check::_AssertTensorDevice(__FILE__, __LINE__, (const char*)__FN__, \
                                      tensor, device, message)
#define _AssertTensorShape2(tensor, shape)                                    \
    tensor_check::_AssertTensorShape(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, shape)
#define _AssertTensorShape3(tensor, shape, message)                           \
    tensor_check::_AssertTensorShape(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, shape, message)
#define _AssertTensorShapeCompatible2(tensor, dynamic_shape) \
    tensor_check::_AssertTensorShapeCompatible(              \
            __FILE__, __LINE__, (const char*)__FN__, tensor, shape)
#define _AssertTensorShapeCompatible3(tensor, dynamic_shape, message) \
    tensor_check::_AssertTensorShapeCompatible(                       \
            __FILE__, __LINE__, (const char*)__FN__, tensor, shape, message)

namespace open3d {
namespace core {
namespace tensor_check {

void _AssertTensorDtype(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const Dtype& dtype,
                        const std::string& message = "");

void _AssertTensorDevice(const char* file,
                         int line,
                         const char* function,
                         const Tensor& tensor,
                         const Device& device,
                         const std::string& message = "");

void _AssertTensorShape(const char* file,
                        int line,
                        const char* function,
                        const Tensor& tensor,
                        const SizeVector& shape,
                        const std::string& message = "");

void _AssertTensorShapeCompatible(const char* file,
                                  int line,
                                  const char* function,
                                  const Tensor& tensor,
                                  const DynamicSizeVector& dynamic_shape,
                                  const std::string& message = "");

}  // namespace tensor_check
}  // namespace core
}  // namespace open3d
