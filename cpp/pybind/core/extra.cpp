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

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"
#include "open3d/utility/Optional.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {

static TensorKey ToTensorKey(int key) { return TensorKey::Index(key); }

static TensorKey ToTensorKey(const py::slice& key) {
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    PySlice_Unpack(key.ptr(), &start, &stop, &step);

    PySliceObject* slice_key = reinterpret_cast<PySliceObject*>(key.ptr());
    return TensorKey::Slice(static_cast<int64_t>(start),
                            static_cast<int64_t>(stop),
                            static_cast<int64_t>(step),
                            py::detail::PyNone_Check(slice_key->start),
                            py::detail::PyNone_Check(slice_key->stop),
                            py::detail::PyNone_Check(slice_key->step));
}

void pybind_core_extra(py::class_<Tensor>& tensor) {
    utility::LogInfo("pybind_core_extra");

    tensor.def("__getitem__", [](const Tensor& tensor, int key) {
        utility::LogInfo("__getitem__ int");
        return tensor.GetItem(ToTensorKey(key));
    });

    tensor.def("__getitem__", [](const Tensor& tensor, const py::slice& key) {
        utility::LogInfo("__getitem__ slice");
        return tensor.GetItem(ToTensorKey(key));
    });

    tensor.def("_getitem", [](const Tensor& tensor, const TensorKey& tk) {
        return tensor.GetItem(tk);
    });

    tensor.def("_getitem_vector",
               [](const Tensor& tensor, const std::vector<TensorKey>& tks) {
                   return tensor.GetItem(tks);
               });

    tensor.def("_setitem",
               [](Tensor& tensor, const TensorKey& tk, const Tensor& value) {
                   return tensor.SetItem(tk, value);
               });

    tensor.def("_setitem_vector",
               [](Tensor& tensor, const std::vector<TensorKey>& tks,
                  const Tensor& value) { return tensor.SetItem(tks, value); });
}

}  // namespace core
}  // namespace open3d
