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

#include "pybind/core/tensor_converter.h"

#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"
#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif
#include "pybind/core/core.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

namespace open3d {
namespace core {

static Tensor CastOptionalDtypeDevice(const Tensor& t,
                                      utility::optional<Dtype> dtype,
                                      utility::optional<Device> device) {
    Tensor t_cast = t;
    if (dtype.has_value()) {
        t_cast = t_cast.To(dtype.value());
    }
    if (device.has_value()) {
        t_cast = t_cast.To(device.value());
    }
    return t_cast;
}

/// Convert Tensor class to py::array (Numpy array).
py::array TensorToPyArray(const Tensor& tensor) {
    if (tensor.GetDevice().GetType() != Device::DeviceType::CPU) {
        utility::LogError(
                "Can only convert CPU Tensor to numpy. Copy Tensor to CPU "
                "before converting to numpy.");
    }
    py::dtype py_dtype =
            py::dtype(pybind_utils::DtypeToArrayFormat(tensor.GetDtype()));
    py::array::ShapeContainer py_shape(tensor.GetShape());
    SizeVector strides = tensor.GetStrides();
    int64_t element_byte_size = tensor.GetDtype().ByteSize();
    for (auto& s : strides) {
        s *= element_byte_size;
    }
    py::array::StridesContainer py_strides(strides);

    // `base_tensor` is a shallow copy of `tensor`. `base_tensor`
    // is on the heap and is owned by py::capsule
    // `base_tensor_capsule`. The capsule is referenced as the
    // "base" of the numpy tensor returned by o3d.Tensor.numpy().
    // When the "base" goes out-of-scope (e.g. when all numpy
    // tensors referencing the base have gone out-of-scope), the
    // deleter is called to free the `base_tensor`.
    //
    // This behavior is important when the origianl `tensor` goes
    // out-of-scope while we still want to keep the data alive.
    // e.g.
    //
    // ```python
    // def get_np_tensor():
    //     o3d_t = o3d.Tensor(...)
    //     return o3d_t.numpy()
    //
    // # Now, `o3d_t` is out-of-scope, but `np_t` still
    // # references the base tensor which references the
    // # underlying data of `o3d_t`. Thus np_t is still valid.
    // # When np_t goes out-of-scope, the underlying data will be
    // # finally freed.
    // np_t = get_np_tensor()
    // ```
    //
    // See:
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    Tensor* base_tensor = new Tensor(tensor);

    // See PyTorch's torch/csrc/Module.cpp
    auto capsule_destructor = [](PyObject* data) {
        Tensor* base_tensor = reinterpret_cast<Tensor*>(
                PyCapsule_GetPointer(data, "open3d::Tensor"));
        if (base_tensor) {
            delete base_tensor;
        } else {
            PyErr_Clear();
        }
    };

    py::capsule base_tensor_capsule(base_tensor, "open3d::Tensor",
                                    capsule_destructor);
    return py::array(py_dtype, py_shape, py_strides, tensor.GetDataPtr(),
                     base_tensor_capsule);
}

Tensor PyArrayToTensor(py::array array, bool inplace) {
    py::buffer_info info = array.request();

    SizeVector shape(info.shape.begin(), info.shape.end());
    SizeVector strides(info.strides.begin(), info.strides.end());
    for (size_t i = 0; i < strides.size(); ++i) {
        strides[i] /= info.itemsize;
    }
    Dtype dtype = pybind_utils::ArrayFormatToDtype(info.format, info.itemsize);
    Device device("CPU:0");

    array.inc_ref();
    std::function<void(void*)> deleter = [array](void*) -> void {
        py::gil_scoped_acquire acquire;
        array.dec_ref();
    };
    auto blob = std::make_shared<Blob>(device, info.ptr, deleter);
    Tensor t_inplace(shape, strides, info.ptr, dtype, blob);

    if (inplace) {
        return t_inplace;
    } else {
        return t_inplace.Clone();
    }
}

Tensor PyListToTensor(const py::list& list,
                      utility::optional<Dtype> dtype,
                      utility::optional<Device> device) {
    py::object numpy = py::module::import("numpy");
    py::array np_array = numpy.attr("array")(list);
    Tensor t = PyArrayToTensor(np_array, false);
    return CastOptionalDtypeDevice(t, dtype, device);
}

Tensor PyTupleToTensor(const py::tuple& tuple,
                       utility::optional<Dtype> dtype,
                       utility::optional<Device> device) {
    py::object numpy = py::module::import("numpy");
    py::array np_array = numpy.attr("array")(tuple);
    Tensor t = PyArrayToTensor(np_array, false);
    return CastOptionalDtypeDevice(t, dtype, device);
}

Tensor DoubleToTensor(double scalar_value,
                      utility::optional<Dtype> dtype,
                      utility::optional<Device> device) {
    Dtype dtype_value = core::Float64;
    if (dtype.has_value()) {
        dtype_value = dtype.value();
    }
    Device device_value("CPU:0");
    if (device.has_value()) {
        device_value = device.value();
    }
    return Tensor(std::vector<double>{scalar_value}, {}, core::Float64,
                  device_value)
            .To(dtype_value);
}

Tensor IntToTensor(int64_t scalar_value,
                   utility::optional<Dtype> dtype,
                   utility::optional<Device> device) {
    Dtype dtype_value = core::Int64;
    if (dtype.has_value()) {
        dtype_value = dtype.value();
    }
    Device device_value("CPU:0");
    if (device.has_value()) {
        device_value = device.value();
    }
    return Tensor(std::vector<int64_t>{scalar_value}, {}, core::Int64,
                  device_value)
            .To(dtype_value);
}

Tensor BoolToTensor(bool scalar_value,
                    utility::optional<Dtype> dtype,
                    utility::optional<Device> device) {
    Dtype dtype_value = core::Bool;
    if (dtype.has_value()) {
        dtype_value = dtype.value();
    }
    Device device_value("CPU:0");
    if (device.has_value()) {
        device_value = device.value();
    }
    return Tensor(std::vector<bool>{scalar_value}, {}, core::Bool, device_value)
            .To(dtype_value);
}

Tensor PyHandleToTensor(const py::handle& handle,
                        utility::optional<Dtype> dtype,
                        utility::optional<Device> device,
                        bool force_copy) {
    // 1) bool
    // 2) int
    // 3) float (double)
    // 4) list
    // 5) tuple
    // 6) numpy.ndarray (value will be copied)
    // 7) Tensor (value will be copied)
    std::string class_name(py::str(handle.get_type()));
    if (class_name == "<class 'bool'>") {
        return BoolToTensor(static_cast<bool>(handle.cast<py::bool_>()), dtype,
                            device);
    } else if (class_name == "<class 'int'>") {
        return IntToTensor(static_cast<int64_t>(handle.cast<py::int_>()), dtype,
                           device);
    } else if (class_name == "<class 'float'>") {
        return DoubleToTensor(static_cast<double>(handle.cast<py::float_>()),
                              dtype, device);
    } else if (class_name == "<class 'list'>") {
        return PyListToTensor(handle.cast<py::list>(), dtype, device);
    } else if (class_name == "<class 'tuple'>") {
        return PyTupleToTensor(handle.cast<py::tuple>(), dtype, device);
    } else if (class_name == "<class 'numpy.ndarray'>") {
        return CastOptionalDtypeDevice(PyArrayToTensor(handle.cast<py::array>(),
                                                       /*inplace=*/!force_copy),
                                       dtype, device);
    } else if (class_name.find("open3d") != std::string::npos &&
               class_name.find("Tensor") != std::string::npos) {
        try {
            Tensor* tensor = handle.cast<Tensor*>();
            if (force_copy) {
                return CastOptionalDtypeDevice(tensor->Clone(), dtype, device);
            } else {
                return CastOptionalDtypeDevice(*tensor, dtype, device);
            }
        } catch (...) {
            utility::LogError("Cannot cast index to Tensor.");
        }
    } else {
        utility::LogError("PyHandleToTensor has invalid input type {}.",
                          class_name);
    }
}

}  // namespace core
}  // namespace open3d
