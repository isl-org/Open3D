// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <vector>

#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"
#include "open3d/utility/Optional.h"
#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif
#include "pybind/core/core.h"
#include "pybind/core/tensor_converter.h"
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

    utility::optional<int64_t> start_opt = None;
    if (!py::detail::PyNone_Check(slice_key->start)) {
        start_opt = static_cast<int64_t>(start);
    }
    utility::optional<int64_t> stop_opt = None;
    if (!py::detail::PyNone_Check(slice_key->stop)) {
        stop_opt = static_cast<int64_t>(stop);
    }
    utility::optional<int64_t> step_opt = None;
    if (!py::detail::PyNone_Check(slice_key->step)) {
        step_opt = static_cast<int64_t>(step);
    }
    return TensorKey::Slice(start_opt, stop_opt, step_opt);
}

static TensorKey ToTensorKey(const py::list& key) {
    Tensor key_tensor = PyTupleToTensor(key);
    if (key_tensor.GetDtype() != core::Bool) {
        key_tensor = key_tensor.To(core::Int64);
    }
    return TensorKey::IndexTensor(key_tensor);
}

static TensorKey ToTensorKey(const py::tuple& key) {
    Tensor key_tensor = PyTupleToTensor(key);
    if (key_tensor.GetDtype() != core::Bool) {
        key_tensor = key_tensor.To(core::Int64);
    }
    return TensorKey::IndexTensor(key_tensor);
}

static TensorKey ToTensorKey(const py::array& key) {
    Tensor key_tensor = PyArrayToTensor(key, /*inplace=*/false);
    if (key_tensor.GetDtype() != core::Bool) {
        key_tensor = key_tensor.To(core::Int64);
    }
    return TensorKey::IndexTensor(key_tensor);
}

static TensorKey ToTensorKey(const Tensor& key_tensor) {
    if (key_tensor.GetDtype() != core::Bool) {
        return TensorKey::IndexTensor(key_tensor.To(core::Int64));
    } else {
        return TensorKey::IndexTensor(key_tensor);
    }
}

/// Convert supported types to TensorKey. Infer types via type name and dynamic
/// casting. Supported types:
/// 1) int
/// 2) slice
/// 3) list
/// 4) tuple
/// 5) numpy.ndarray
/// 6) Tensor
static TensorKey PyHandleToTensorKey(const py::handle& item) {
    // Infer types from type name and dynamic casting.
    // See: https://github.com/pybind/pybind11/issues/84.
    std::string class_name(py::str(item.get_type()));
    if (class_name == "<class 'int'>") {
        return ToTensorKey(static_cast<int64_t>(item.cast<py::int_>()));
    } else if (class_name == "<class 'slice'>") {
        return ToTensorKey(item.cast<py::slice>());
    } else if (class_name == "<class 'list'>") {
        return ToTensorKey(item.cast<py::list>());
    } else if (class_name == "<class 'tuple'>") {
        return ToTensorKey(item.cast<py::tuple>());
    } else if (class_name == "<class 'numpy.ndarray'>") {
        return ToTensorKey(item.cast<py::array>());
    } else if (class_name.find("open3d") != std::string::npos &&
               class_name.find("Tensor") != std::string::npos) {
        try {
            Tensor* tensor = item.cast<Tensor*>();
            return ToTensorKey(*tensor);
        } catch (...) {
            utility::LogError("Cannot cast index to Tensor.");
        }
    } else {
        utility::LogError("PyHandleToTensorKey has invalid key type {}.",
                          class_name);
    }
}

static void pybind_getitem(py::class_<Tensor>& tensor) {
    tensor.def("__getitem__", [](const Tensor& tensor, bool key) {
        return tensor.GetItem(ToTensorKey(Tensor::Init(key)));
    });

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
        std::vector<TensorKey> tks;
        for (const py::handle item : key) {
            tks.push_back(PyHandleToTensorKey(item));
        }
        return tensor.GetItem(tks);
    });
}

static void pybind_setitem(py::class_<Tensor>& tensor) {
    tensor.def("__setitem__", [](Tensor& tensor, bool key,
                                 const py::handle& value) {
        return tensor.SetItem(
                ToTensorKey(Tensor::Init(key)),
                PyHandleToTensor(value, tensor.GetDtype(), tensor.GetDevice(),
                                 /*force_copy=*/false));
    });

    tensor.def("__setitem__", [](Tensor& tensor, int key,
                                 const py::handle& value) {
        return tensor.SetItem(
                ToTensorKey(key),
                PyHandleToTensor(value, tensor.GetDtype(), tensor.GetDevice(),
                                 /*force_copy=*/false));
    });

    tensor.def("__setitem__", [](Tensor& tensor, const py::slice& key,
                                 const py::handle& value) {
        return tensor.SetItem(
                ToTensorKey(key),
                PyHandleToTensor(value, tensor.GetDtype(), tensor.GetDevice(),
                                 /*force_copy=*/false));
    });

    tensor.def("__setitem__", [](Tensor& tensor, const py::array& key,
                                 const py::handle& value) {
        return tensor.SetItem(
                ToTensorKey(key),
                PyHandleToTensor(value, tensor.GetDtype(), tensor.GetDevice(),
                                 /*force_copy=*/false));
    });

    tensor.def("__setitem__", [](Tensor& tensor, const Tensor& key,
                                 const py::handle& value) {
        return tensor.SetItem(
                ToTensorKey(key),
                PyHandleToTensor(value, tensor.GetDtype(), tensor.GetDevice(),
                                 /*force_copy=*/false));
    });

    // List is interpreted as one TensorKey object, which calls
    // Tensor::SetItem(const TensorKey&, xxx).
    // E.g. a[[3, 4, 5]] = xxx is a list. It indices the first dimension of a.
    // E.g. a[(3, 4, 5)] = xxx does very different things. It indices the first
    // three dimensions of a.
    tensor.def("__setitem__", [](Tensor& tensor, const py::list& key,
                                 const py::handle& value) {
        return tensor.SetItem(
                ToTensorKey(key),
                PyHandleToTensor(value, tensor.GetDtype(), tensor.GetDevice(),
                                 /*force_copy=*/false));
    });

    // Tuple is interpreted as a vector TensorKey objects, which calls
    // Tensor::SetItem(const std::vector<TensorKey>&, xxx).
    // E.g. a[1:2, [3, 4, 5], 3:10] = xxx results in a tuple of size 3.
    tensor.def("__setitem__", [](Tensor& tensor, const py::tuple& key,
                                 const py::handle& value) {
        std::vector<TensorKey> tks;
        for (const py::handle item : key) {
            tks.push_back(PyHandleToTensorKey(item));
        }
        return tensor.SetItem(tks, PyHandleToTensor(value, tensor.GetDtype(),
                                                    tensor.GetDevice(),
                                                    /*force_copy=*/false));
    });
}

void pybind_core_tensor_accessor(py::class_<Tensor>& tensor) {
    pybind_getitem(tensor);
    pybind_setitem(tensor);
}

}  // namespace core
}  // namespace open3d
