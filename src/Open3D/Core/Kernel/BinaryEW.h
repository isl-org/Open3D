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

#include <unordered_set>

#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

namespace open3d {
namespace kernel {

enum class BinaryEWOpCode {
    Add,
    Sub,
    Mul,
    Div,
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

extern const std::unordered_set<BinaryEWOpCode, utility::hash_enum_class::hash>
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

inline void Add(const Tensor& lhs, const Tensor& rhs, Tensor& dst) {
    BinaryEW(lhs, rhs, dst, BinaryEWOpCode::Add);
}

inline void Sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst) {
    BinaryEW(lhs, rhs, dst, BinaryEWOpCode::Sub);
}

inline void Mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) {
    BinaryEW(lhs, rhs, dst, BinaryEWOpCode::Mul);
}

inline void Div(const Tensor& lhs, const Tensor& rhs, Tensor& dst) {
    BinaryEW(lhs, rhs, dst, BinaryEWOpCode::Div);
}

}  // namespace kernel
}  // namespace open3d
