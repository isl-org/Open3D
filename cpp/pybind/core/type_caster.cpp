// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/core/type_caster.h"

#include "pybind/core/tensor_converter.h"

namespace pybind11 {
namespace detail {

bool type_caster<open3d::core::Tensor>::load(handle src, bool convert) {
    using base = type_caster_base<open3d::core::Tensor>;
    if (this->base::load(src, convert)) {
        return true;
    }

    if (convert) {
        if (py::isinstance<py::bool_>(src) || py::isinstance<py::int_>(src) ||
            py::isinstance<py::float_>(src) || py::isinstance<py::list>(src) ||
            py::isinstance<py::tuple>(src) || py::isinstance<py::array>(src)) {
            holder_ = std::make_unique<open3d::core::Tensor>(
                    open3d::core::PyHandleToTensor(src));
            value = holder_.get();
            return true;
        }
    }

    return false;
}

bool type_caster<open3d::core::Device>::load(handle src, bool convert) {
    if (type_caster_base<open3d::core::Device>::load(src, convert)) {
        return true;
    }
    if (convert && py::isinstance<py::str>(src)) {
        holder_ = std::make_unique<open3d::core::Device>(
                py::cast<std::string>(src));
        value = holder_.get();
        return true;
    }
    return false;
}

}  // namespace detail
}  // namespace pybind11
