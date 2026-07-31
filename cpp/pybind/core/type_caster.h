// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "open3d/core/Device.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {
class Tensor;
}
}  // namespace open3d

// Type casters for implicit Python conversions. Include in each compilation
// unit that binds APIs using these types (see open3d_pybind.h).
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

template <>
struct type_caster<open3d::core::Device>
    : public type_caster_base<open3d::core::Device> {
public:
    bool load(handle src, bool convert);

private:
    std::unique_ptr<open3d::core::Device> holder_;
};

}  // namespace detail
}  // namespace pybind11
