// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/core/tensor_type_caster.h"

#include "pybind/core/tensor_converter.h"

namespace pybind11 {
namespace detail {

bool type_caster<open3d::core::Tensor>::load(handle src, bool convert) {
    using base = type_caster_base<open3d::core::Tensor>;
    if (this->base::load(src, convert)) {
        return true;
    }

    if (convert) {
        std::string class_name(py::str(src.get_type()));
        if (class_name == "<class 'bool'>" || class_name == "<class 'int'>" ||
            class_name == "<class 'float'>" || class_name == "<class 'list'>" ||
            class_name == "<class 'tuple'>" ||
            class_name == "<class 'numpy.ndarray'>") {
            holder_ = std::make_unique<open3d::core::Tensor>(
                    open3d::core::PyHandleToTensor(src));
            value = holder_.get();
            return true;
        }
    }

    return false;
}

}  // namespace detail
}  // namespace pybind11
