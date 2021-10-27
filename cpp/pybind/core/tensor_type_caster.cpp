// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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
