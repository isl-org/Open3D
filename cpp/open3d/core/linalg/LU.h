// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

// See documentation for `core::Tensor::LUIpiv`.
void LUIpiv(const Tensor& A, Tensor& ipiv, Tensor& output);

// See documentation for `core::Tensor::LU`.
void LU(const Tensor& A,
        Tensor& permutation,
        Tensor& lower,
        Tensor& upper,
        const bool permute_l = false);

}  // namespace core
}  // namespace open3d
