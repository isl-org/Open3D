// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <unordered_set>

#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {
namespace kernel {

enum class ReductionOpCode {
    Sum,
    Prod,
    Min,
    Max,
    ArgMin,
    ArgMax,
    All,
    Any,
};

static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_regular_reduce_ops = {
                ReductionOpCode::Sum,
                ReductionOpCode::Prod,
                ReductionOpCode::Min,
                ReductionOpCode::Max,
};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_arg_reduce_ops = {
                ReductionOpCode::ArgMin,
                ReductionOpCode::ArgMax,
};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_boolean_reduce_ops = {
                ReductionOpCode::All,
                ReductionOpCode::Any,
};

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
}  // namespace core
}  // namespace open3d
