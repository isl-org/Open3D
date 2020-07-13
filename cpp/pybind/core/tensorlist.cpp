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

#include "open3d/core/TensorList.h"

#include <vector>

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {

void pybind_core_tensorlist(py::module& m) {
    py::class_<core::TensorList> tensorlist(
            m, "TensorList",
            "A TensorList is an extendable tensor at the 0-th dimension.");

    // Constructors.
    tensorlist.def(
            py::init([](const core::SizeVector& element_shape,
                        const core::Dtype& dtype, const core::Device& device) {
                return new core::TensorList(element_shape, dtype, device);
            }),
            "element_shape"_a, "dtype"_a, "device"_a);
    tensorlist.def(py::init([](const std::vector<core::Tensor>& tensors) {
                       return new core::TensorList(tensors);
                   }),
                   "tensors"_a);
    tensorlist.def(py::init([](const core::TensorList& other) {
                       return new core::TensorList(other);
                   }),
                   "other"_a);

    // Factory function.
    tensorlist.def_static("from_tensor", &core::TensorList::FromTensor,
                          "tensor"_a, "inplace"_a = false);

    // Copiers.
    tensorlist.def("shallow_copy_from", &core::TensorList::ShallowCopyFrom);
    tensorlist.def("copy_from", &core::TensorList::CopyFrom);
    tensorlist.def("copy", &core::TensorList::Copy);

    // Accessors.
    tensorlist.def("__getitem__", [](core::TensorList& tl, int64_t index) {
        return tl[index];
    });
    tensorlist.def("__setitem__",
                   [](core::TensorList& tl, int64_t index,
                      const core::Tensor& value) { tl[index] = value; });
    tensorlist.def("as_tensor",
                   [](const core::TensorList& tl) { return tl.AsTensor(); });
    tensorlist.def("__repr__",
                   [](const core::TensorList& tl) { return tl.ToString(); });
    tensorlist.def("__str__",
                   [](const core::TensorList& tl) { return tl.ToString(); });

    // Manipulations.
    tensorlist.def("push_back", &core::TensorList::PushBack);
    tensorlist.def("resize", &core::TensorList::Resize);
    tensorlist.def("extend", &core::TensorList::Extend);
    tensorlist.def("__iadd__", &core::TensorList::operator+=);
    tensorlist.def("__add__", &core::TensorList::operator+);
    tensorlist.def_static("concat", &core::TensorList::Concatenate);

    // Properties.
    tensorlist.def_property_readonly("size", &core::TensorList::GetSize);
    tensorlist.def_property_readonly("element_shape",
                                     &core::TensorList::GetElementShape);
    tensorlist.def_property_readonly("dtype", &core::TensorList::GetDtype);
    tensorlist.def_property_readonly("device", &core::TensorList::GetDevice);
    tensorlist.def_property_readonly("is_resizable",
                                     &core::TensorList::IsResizable);
}

}  // namespace open3d
