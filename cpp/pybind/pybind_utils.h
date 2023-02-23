// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace pybind_utils {

core::Dtype ArrayFormatToDtype(const std::string& format, size_t byte_size);

std::string DtypeToArrayFormat(const core::Dtype& dtype);

}  // namespace pybind_utils

}  // namespace open3d
