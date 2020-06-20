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

    tensorlist
            .def(py::init(
                    [](const core::SizeVector& shape, const core::Dtype& dtype,
                       const core::Device& device, const int64_t& size = 0) {
                        core::TensorList tl =
                                core::TensorList(shape, dtype, device, size);
                        return tl;
                    }))
            .def("shallow_copy_from", &core::TensorList::ShallowCopyFrom)
            .def("copy_from", &core::TensorList::CopyFrom)
            // Construct from existing tensors with compatible shapes
            .def("from_tensors",
                 [](const std::vector<core::Tensor>& tensors,
                    const core::Device& device) {
                     core::TensorList tl = core::TensorList(tensors, device);
                     return tl;
                 })
            // Construct from existing internal tensor with at least one valid
            // dimension
            .def_static("from_tensor", &core::TensorList::FromTensor)
            .def("tensor",
                 [](const core::TensorList& tl) { return tl.AsTensor(); })
            .def("push_back",
                 [](core::TensorList& tl, const core::Tensor& tensor) {
                     return tl.PushBack(tensor);
                 })
            .def("resize",
                 [](core::TensorList& tl, int64_t n) { return tl.Resize(n); })
            .def("extend",
                 [](core::TensorList& tl_a, const core::TensorList& tl_b) {
                     return tl_a.Extend(tl_b);
                 })
            .def("size",
                 [](const core::TensorList& tl) { return tl.GetSize(); })

            .def("_getitem",
                 [](core::TensorList& tl, int64_t index) { return tl[index]; })
            .def("_setitem",
                 [](core::TensorList& tl, int64_t index,
                    const core::Tensor& value) { tl[index].SetItem(value); })

            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("__repr__",
                 [](const core::TensorList& tl) { return tl.ToString(); })
            .def_static("concat", [](const core::TensorList& tl_a,
                                     const core::TensorList& tl_b) {
                return core::TensorList::Concatenate(tl_a, tl_b);
            });

    tensorlist.def_property_readonly("shape", &core::TensorList::GetShape);
    tensorlist.def_property_readonly("dtype", &core::TensorList::GetDtype);
    tensorlist.def_property_readonly("device", &core::TensorList::GetDevice);
}

}  // namespace open3d
