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

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"

#ifdef _WIN32
#define __FN__ __FUNCSIG__
#else
#define __FN__ __PRETTY_FUNCTION__
#endif

#define OPEN3D_GET_4TH_ARG(arg0, arg1, arg2, arg3, ...) arg3

#define AssertTensorDtype2(tensor, dtype)                                     \
    tensor_check::_AssertTensorDtype(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, dtype)
#define AssertTensorDtype3(tensor, dtype, message)                            \
    tensor_check::_AssertTensorDtype(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, dtype, message)
#define AssertTensorDtype(...)                                                \
    OPEN3D_GET_4TH_ARG(__VA_ARGS__, AssertTensorDtype3, AssertTensorDtype2, ) \
    (__VA_ARGS__)

#define AssertTensorDevice2(tensor, device)                                    \
    tensor_check::_AssertTensorDevice(__FILE__, __LINE__, (const char*)__FN__, \
                                      tensor, device)
#define AssertTensorDevice3(tensor, device, message)                           \
    tensor_check::_AssertTensorDevice(__FILE__, __LINE__, (const char*)__FN__, \
                                      tensor, device, message)
#define AssertTensorDevice(...)                          \
    OPEN3D_GET_4TH_ARG(__VA_ARGS__, AssertTensorDevice3, \
                       AssertTensorDevice2, )            \
    (__VA_ARGS__)

#define AssertTensorShape2(tensor, shape)                                     \
    tensor_check::_AssertTensorShape(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, shape)
#define AssertTensorShape3(tensor, shape, message)                            \
    tensor_check::_AssertTensorShape(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, shape, message)
#define AssertTensorShape(...)                                                \
    OPEN3D_GET_4TH_ARG(__VA_ARGS__, AssertTensorShape3, AssertTensorShape2, ) \
    (__VA_ARGS__)

#define AssertTensorShapeCompatible2(tensor, dynamic_shape) \
    tensor_check::_AssertTensorShapeCompatible(             \
            __FILE__, __LINE__, (const char*)__FN__, tensor, shape)
#define AssertTensorShapeCompatible3(tensor, dynamic_shape, message) \
    tensor_check::_AssertTensorShapeCompatible(                      \
            __FILE__, __LINE__, (const char*)__FN__, tensor, shape, message)
#define AssertTensorShapeCompatible(...)                          \
    OPEN3D_GET_4TH_ARG(__VA_ARGS__, AssertTensorShapeCompatible3, \
                       AssertTensorShapeCompatible2, )            \
    (__VA_ARGS__)

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
