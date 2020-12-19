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

static TensorKey ToTensorKey(const py::list& key) {
    Tensor key_tensor = PyTupleToTensor(key);
    if (key_tensor.GetDtype() != Dtype::Bool) {
        key_tensor = key_tensor.To(Dtype::Int64, /*copy=*/false);
    }
    return TensorKey::IndexTensor(key_tensor);
}

static TensorKey ToTensorKey(const py::tuple& key) {
    Tensor key_tensor = PyTupleToTensor(key);
    if (key_tensor.GetDtype() != Dtype::Bool) {
        key_tensor = key_tensor.To(Dtype::Int64, /*copy=*/false);
    }
    return TensorKey::IndexTensor(key_tensor);
}

static TensorKey ToTensorKey(const py::array& key) {
    Tensor key_tensor = PyArrayToTensor(key, /*inplace=*/false);
    if (key_tensor.GetDtype() != Dtype::Bool) {
        key_tensor = key_tensor.To(Dtype::Int64);
    }
    return TensorKey::IndexTensor(key_tensor);
}

static TensorKey ToTensorKey(const Tensor& key_tensor) {
    if (key_tensor.GetDtype() != Dtype::Bool) {
        return TensorKey::IndexTensor(
                key_tensor.To(Dtype::Int64, /*copy=*/false));
    } else {
        return TensorKey::IndexTensor(key_tensor);
    }
}

void pybind_core_extra(py::class_<Tensor>& tensor) {
    utility::LogInfo("pybind_core_extra");

    tensor.def("__getitem__", [](const Tensor& tensor, int key) {
        return tensor.GetItem(ToTensorKey(key));
    });

    tensor.def("__getitem__", [](const Tensor& tensor, const py::slice& key) {
        return tensor.GetItem(ToTensorKey(key));
    });

    tensor.def("__getitem__", [](const Tensor& tensor, const py::array& key) {
        return tensor.GetItem(ToTensorKey(key));
    });

    tensor.def("__getitem__", [](const Tensor& tensor, const Tensor& key) {
        utility::LogInfo("getitem tensor");
        return tensor.GetItem(ToTensorKey(key));
    });

    // List is interpreted as one TensorKey object, which calls
    // Tensor::GetItem(const TensorKey&).
    // E.g. a[[3, 4, 5]] is a list. It indices the first dimension of a.
    // E.g. a[(3, 4, 5)] does very different things. It indices the first three
    //      dimensions of a.
    tensor.def("__getitem__", [](const Tensor& tensor, const py::list& key) {
        return tensor.GetItem(ToTensorKey(key));
    });

    // Tuple is interpreted as a vector TensorKey objects, which calls
    // Tensor::GetItem(const std::vector<TensorKey>&).
    // E.g. a[1:2, [3, 4, 5], 3:10] results in a tuple of size 3.
    tensor.def("__getitem__", [](const Tensor& tensor, const py::tuple& key) {
        for (const auto& item : key) {
            // if (py::slice(item).check()) {
            //     utility::LogInfo("got slice");
            // } else {
            //     utility::LogInfo("got something else");
            // }
            utility::LogInfo("type: {}", item.get_type().str());
        }
        utility::LogError("tuple not supported.");
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
