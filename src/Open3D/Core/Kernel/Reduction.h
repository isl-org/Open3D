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

#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"

#include <unordered_set>

namespace open3d {
namespace kernel {

enum class ReductionOpCode { Sum, Prod, Min, Max, ArgMin, ArgMax };

static const std::unordered_set<ReductionOpCode, utility::hash_enum_class::hash>
        regular_reduce_ops = {ReductionOpCode::Sum, ReductionOpCode::Prod,
                              ReductionOpCode::Min, ReductionOpCode::Max};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class::hash>
        arg_reduce_ops = {ReductionOpCode::ArgMin, ReductionOpCode::ArgMax};

void Reduction(const Tensor& src,
               Tensor& dst,
               const SizeVector& dims,
               bool keepdim,
               ReductionOpCode op_code);

void ReductionCPU(const Tensor& src,
                  Tensor& dst,
                  const SizeVector& dims,
                  bool keepdim,
                  ReductionOpCode op_code);

#ifdef BUILD_CUDA_MODULE
void ReductionCUDA(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code);
#endif

}  // namespace kernel
}  // namespace open3d
