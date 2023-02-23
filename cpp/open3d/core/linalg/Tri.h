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

// See documentation for `core::Tensor::Triu`.
void Triu(const Tensor& A, Tensor& output, const int diagonal = 0);

// See documentation for `core::Tensor::Tril`.
void Tril(const Tensor& A, Tensor& output, const int diagonal = 0);

// See documentation for `core::Tensor::Triul`.
void Triul(const Tensor& A,
           Tensor& upper,
           Tensor& lower,
           const int diagonal = 0);

}  // namespace core
}  // namespace open3d
