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

#include <unordered_set>

#include "open3d/core/Tensor.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

enum class BinaryEWOpCode {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
};

extern const std::unordered_set<BinaryEWOpCode, utility::hash_enum_class>
        s_boolean_binary_ew_op_codes;

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              Tensor& dst,
              BinaryEWOpCode op_code);

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code);

#ifdef BUILD_CUDA_MODULE
void BinaryEWCUDA(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  BinaryEWOpCode op_code);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace open3d
