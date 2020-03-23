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

#include "Open3D/Core/TensorList.h"

#include <vector>

#include "Open3D/Core/Blob.h"
#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Core/Dispatch.h"
#include "Open3D/Core/Dtype.h"
#include "Open3D/Core/SizeVector.h"
#include "Open3D/Core/Tensor.h"
#include "open3d_pybind/core/container.h"
#include "open3d_pybind/docstring.h"
#include "open3d_pybind/open3d_pybind.h"
#include "open3d_pybind/pybind_utils.h"
using namespace open3d;

void pybind_core_tensorlist(py::module& m) {
    py::class_<TensorList> tensorlist(
            m, "TensorList",
            "A TensorList is an extendable tensor at the 0-th dimension.");

    tensorlist
            .def(py::init([](const SizeVector& shape, const Dtype& dtype,
                             const Device& device, const int64_t& size = 0) {
                TensorList tl = TensorList(shape, dtype, device, size);
                return tl;
            }))
            .def("shallow_copy_from", &TensorList::ShallowCopyFrom)
            .def("copy_from", &TensorList::CopyFrom)
            // Construct from existing tensors with compatible shapes
            .def("from_tensors",
                 [](const std::vector<Tensor>& tensors, const Device& device) {
                     TensorList tl = TensorList(tensors, device);
                     return tl;
                 })
            // Construct from existing internal tensor with at least one valid
            // dimension
            .def_static("from_tensor", &TensorList::FromTensor)
            .def("tensor", [](const TensorList& tl) { return tl.AsTensor(); })
            .def("push_back",
                 [](TensorList& tl, const Tensor& tensor) {
                     return tl.PushBack(tensor);
                 })
            .def("resize",
                 [](TensorList& tl, int64_t n) { return tl.Resize(n); })
            .def("extend",
                 [](TensorList& tl_a, const TensorList& tl_b) {
                     return tl_a.Extend(tl_b);
                 })
            .def("size", [](const TensorList& tl) { return tl.GetSize(); })

            .def("_getitem",
                 [](TensorList& tl, int64_t index) { return tl[index]; })
            .def("_setitem",
                 [](TensorList& tl, int64_t index, const Tensor& value) {
                     tl[index].SetItem(value);
                 })

            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("__repr__", [](const TensorList& tl) { return tl.ToString(); })
            .def_static("concat",
                        [](const TensorList& tl_a, const TensorList& tl_b) {
                            return TensorList::Concatenate(tl_a, tl_b);
                        });
}
