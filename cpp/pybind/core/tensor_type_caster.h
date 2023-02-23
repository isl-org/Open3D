// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {
class Tensor;
}
}  // namespace open3d

// Define type caster allowing implicit conversion to Tensor from common types.
// Needs to be included in each compilation unit.
namespace pybind11 {
namespace detail {
template <>
struct type_caster<open3d::core::Tensor>
    : public type_caster_base<open3d::core::Tensor> {
public:
    bool load(handle src, bool convert);

private:
    std::unique_ptr<open3d::core::Tensor> holder_;
};

}  // namespace detail
}  // namespace pybind11
