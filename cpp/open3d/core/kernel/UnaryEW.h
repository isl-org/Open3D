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

#pragma once

#include "open3d/core/Tensor.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace core {
namespace kernel {

enum class UnaryEWOpCode {
    Sqrt,
    Sin,
    Cos,
    Neg,
    Exp,
    Abs,
    Floor,
    Ceil,
    Round,
    Trunc,
    LogicalNot
};

void UnaryEW(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);

void UnaryEWCPU(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);

#ifdef BUILD_CUDA_MODULE
void UnaryEWCUDA(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code);
#endif

// Copy is separated from other unary ops since it support cross-device copy and
// dtype casting.
void Copy(const Tensor& src, Tensor& dst);

void CopyCPU(const Tensor& src, Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void CopyCUDA(const Tensor& src, Tensor& dst);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace open3d
