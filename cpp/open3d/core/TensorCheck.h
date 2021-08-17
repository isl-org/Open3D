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

#define AssertTensorDtype2Args(tensor, dtype)                                 \
    tensor_check::_AssertTensorDtype(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, dtype)
#define AssertTensorDtype3Args(tensor, dtype, message)                        \
    tensor_check::_AssertTensorDtype(__FILE__, __LINE__, (const char*)__FN__, \
                                     tensor, dtype, message)
#define AssertTensorDtype(...)                              \
    OPEN3D_GET_4TH_ARG(__VA_ARGS__, AssertTensorDtype3Args, \
                       AssertTensorDtype2Args, )            \
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

}
}  // namespace core
}  // namespace open3d
