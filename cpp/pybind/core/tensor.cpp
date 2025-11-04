// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/Tensor.h"

#include <optional>
#include <vector>

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/DLPack.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/Scalar.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/TensorKey.h"
#include "pybind/core/core.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/core/tensor_type_caster.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

#define CONST_ARG const
#define NON_CONST_ARG

#define BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(py_name, cpp_name, self_const) \
    tensor.def(#py_name, [](self_const Tensor& self, const Tensor& other) {  \
        return self.cpp_name(other);                                         \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, float value) {                     \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, double value) {                    \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, int8_t value) {                    \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, int16_t value) {                   \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, int32_t value) {                   \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, int64_t value) {                   \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, uint8_t value) {                   \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, uint16_t value) {                  \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, uint32_t value) {                  \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, uint64_t value) {                  \
        return self.cpp_name(Scalar(value));                                 \
    });                                                                      \
    tensor.def(#py_name, [](Tensor& self, bool value) {                      \
        return self.cpp_name(Scalar(value));                                 \
    });

#define BIND_CLIP_SCALAR(py_name, cpp_name, self_const)                        \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, float min_v, float max_v) {         \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, double min_v, double max_v) {       \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, int8_t min_v, int8_t max_v) {       \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, int16_t min_v, int16_t max_v) {     \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, int32_t min_v, int32_t max_v) {     \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, int64_t min_v, int64_t max_v) {     \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, uint8_t min_v, uint8_t max_v) {     \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, uint16_t min_v, uint16_t max_v) {   \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, uint32_t min_v, uint32_t max_v) {   \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name,                                                       \
               [](self_const Tensor& self, uint64_t min_v, uint64_t max_v) {   \
                   return self.cpp_name(min_v, max_v);                         \
               });                                                             \
    tensor.def(#py_name, [](self_const Tensor& self, bool min_v, bool max_v) { \
        return self.cpp_name(min_v, max_v);                                    \
    });

#define BIND_BINARY_R_OP_ALL_DTYPES(py_name, cpp_name)                    \
    tensor.def(#py_name, [](const Tensor& self, float value) {            \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, double value) {           \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, int8_t value) {           \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, int16_t value) {          \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, int32_t value) {          \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, int64_t value) {          \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, uint8_t value) {          \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, uint16_t value) {         \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, uint32_t value) {         \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, uint64_t value) {         \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });                                                                   \
    tensor.def(#py_name, [](const Tensor& self, bool value) {             \
        return Tensor::Full({}, value, self.GetDtype(), self.GetDevice()) \
                .cpp_name(self);                                          \
    });

#define BIND_REDUCTION_OP(py_name, cpp_name)                               \
    tensor.def(                                                            \
            #py_name,                                                      \
            [](const Tensor& tensor, const std::optional<SizeVector>& dim, \
               bool keepdim) {                                             \
                SizeVector reduction_dims;                                 \
                if (dim.has_value()) {                                     \
                    reduction_dims = dim.value();                          \
                } else {                                                   \
                    for (int64_t i = 0; i < tensor.NumDims(); i++) {       \
                        reduction_dims.push_back(i);                       \
                    }                                                      \
                }                                                          \
                return tensor.cpp_name(reduction_dims, keepdim);           \
            },                                                             \
            "dim"_a = py::none(), "keepdim"_a = false);

// TODO (rishabh): add this behavior to the cpp implementation. Refer to the
// Tensor::Any and Tensor::All ops.
#define BIND_REDUCTION_OP_NO_KEEPDIM(py_name, cpp_name)                      \
    tensor.def(                                                              \
            #py_name,                                                        \
            [](const Tensor& tensor, const std::optional<SizeVector>& dim) { \
                SizeVector reduction_dims;                                   \
                if (dim.has_value()) {                                       \
                    reduction_dims = dim.value();                            \
                } else {                                                     \
                    for (int64_t i = 0; i < tensor.NumDims(); i++) {         \
                        reduction_dims.push_back(i);                         \
                    }                                                        \
                }                                                            \
                return tensor.cpp_name(reduction_dims);                      \
            },                                                               \
            "dim"_a = py::none());

namespace open3d {
namespace core {

const std::unordered_map<std::string, std::string> argument_docs = {
        {"dtype", "Data type for the Tensor."},
        {"device", "Compute device to store and operate on the Tensor."},
        {"shape", "List of Tensor dimensions."},
        {"fill_value", "Scalar value to initialize all elements with."},
        {"scalar_value", "Initial value for the single element tensor."},
        {"copy",
         "If true, a new tensor is always created; if false, the copy is "
         "avoided when the original tensor already has the targeted dtype."}};

template <typename T>
static std::vector<T> ToFlatVector(
        py::array_t<T, py::array::c_style | py::array::forcecast> np_array) {
    py::buffer_info info = np_array.request();
    T* start = static_cast<T*>(info.ptr);
    return std::vector<T>(start, start + info.size);
}

template <typename func_t>
static void BindTensorCreation(py::module& m,
                               py::class_<Tensor>& tensor,
                               const std::string& py_name,
                               func_t cpp_func) {
    tensor.def_static(
            py_name.c_str(),
            [cpp_func](const SizeVector& shape, std::optional<Dtype> dtype,
                       std::optional<Device> device) {
                return cpp_func(
                        shape,
                        dtype.has_value() ? dtype.value() : core::Float32,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "Create Tensor with a given shape.", "shape"_a,
            "dtype"_a = py::none(), "device"_a = py::none());

    docstring::ClassMethodDocInject(m, "Tensor", py_name, argument_docs);
}

template <typename T>
static void BindTensorFullCreation(py::module& m, py::class_<Tensor>& tensor) {
    tensor.def_static(
            "full",
            [](const SizeVector& shape, T fill_value,
               std::optional<Dtype> dtype, std::optional<Device> device) {
                return Tensor::Full<T>(
                        shape, fill_value,
                        dtype.has_value() ? dtype.value() : core::Float32,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "shape"_a, "fill_value"_a, "dtype"_a = py::none(),
            "device"_a = py::none());
}

void pybind_core_tensor_declarations(py::module& m) {
    py::class_<Tensor> tensor(
            m, "Tensor",
            "A Tensor is a view of a data Blob with shape, stride, data_ptr.");
    m.attr("capsule") = py::module_::import("typing").attr("Any");
    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#array_api.array.__dlpack_device__
    py::native_enum<DLDeviceType>(m, "DLDeviceType", "enum.Enum")
            .value("CPU", DLDeviceType::kDLCPU)
            .value("CUDA", DLDeviceType::kDLCUDA)
            .value("CPU_PINNED", DLDeviceType::kDLCUDAHost)
            .value("OPENCL", DLDeviceType::kDLOpenCL)
            .value("VULKAN", DLDeviceType::kDLVulkan)
            .value("METAL", DLDeviceType::kDLMetal)
            .value("VPI", DLDeviceType::kDLVPI)
            .value("ROCM", DLDeviceType::kDLROCM)
            .value("CUDA_MANAGED", DLDeviceType::kDLCUDAManaged)
            .value("ONE_API", DLDeviceType::kDLOneAPI)
            .finalize();
}

namespace {
Tensor maybeCopyTensor(const Tensor& data,
                       std::optional<std::pair<DLDeviceType, int>>
                               optional_dl_device = std::nullopt,
                       std::optional<bool> copy = std::nullopt) {
    bool force_copy = copy.has_value() && copy.value();
    bool force_move = copy.has_value() && !copy.value();
    if (optional_dl_device.has_value()) {
        Device to_device;
        switch (optional_dl_device.value().first) {
            case DLDeviceType::kDLCPU:
                to_device = Device("CPU:0");
                break;
            case DLDeviceType::kDLCUDA:
                to_device = Device(fmt::format(
                        "CUDA:{}", optional_dl_device.value().second));
                break;
            case DLDeviceType::kDLOneAPI:
                to_device = Device(fmt::format(
                        "SYCL:{}", optional_dl_device.value().second));
                break;
            default:
                utility::LogError("DLPack: unsupported device type {}",
                                  int(optional_dl_device.value().first));
        }
        if (to_device != data.GetDevice()) {
            if (!force_move)
                utility::LogError(
                        " [DLPack] Cannot move (i.e. copy=False) tensor from "
                        "{} to {} without copying.",
                        data.GetDevice().ToString(), to_device.ToString());
            return data.To(to_device);
        }
    }
    if (force_copy) {
        return data.Clone();
    }
    return data;
}
}  // namespace

void pybind_core_tensor_definitions(py::module& m) {
    auto tensor = static_cast<py::class_<Tensor>>(m.attr("Tensor"));
    // o3c.Tensor(np.array([[0, 1, 2], [3, 4, 5]]), dtype=None, device=None).
    tensor.def(
            py::init([](const py::array& np_array, std::optional<Dtype> dtype,
                        std::optional<Device> device) {
                Tensor t = PyArrayToTensor(np_array, /*inplace=*/false);
                if (dtype.has_value()) {
                    t = t.To(dtype.value());
                }
                if (device.has_value()) {
                    t = t.To(device.value());
                }
                return t;
            }),
            "Initialize Tensor from a Numpy array.", "np_array"_a,
            "dtype"_a = py::none(), "device"_a = py::none());

    // o3c.Tensor(True, dtype=None, device=None).
    // Default to Bool, CPU:0.
    tensor.def(py::init([](bool scalar_value, std::optional<Dtype> dtype,
                           std::optional<Device> device) {
                   return BoolToTensor(scalar_value, dtype, device);
               }),
               "scalar_value"_a, "dtype"_a = py::none(),
               "device"_a = py::none());

    // o3c.Tensor(1, dtype=None, device=None).
    // Default to Int64, CPU:0.
    tensor.def(py::init([](int64_t scalar_value, std::optional<Dtype> dtype,
                           std::optional<Device> device) {
                   return IntToTensor(scalar_value, dtype, device);
               }),
               "scalar_value"_a, "dtype"_a = py::none(),
               "device"_a = py::none());

    // o3c.Tensor(3.14, dtype=None, device=None).
    // Default to Float64, CPU:0.
    tensor.def(py::init([](double scalar_value, std::optional<Dtype> dtype,
                           std::optional<Device> device) {
                   return DoubleToTensor(scalar_value, dtype, device);
               }),
               "scalar_value"_a, "dtype"_a = py::none(),
               "device"_a = py::none());

    // o3c.Tensor([[0, 1, 2], [3, 4, 5]], dtype=None, device=None).
    tensor.def(py::init([](const py::list& shape, std::optional<Dtype> dtype,
                           std::optional<Device> device) {
                   return PyListToTensor(shape, dtype, device);
               }),
               "Initialize Tensor from a nested list.", "shape"_a,
               "dtype"_a = py::none(), "device"_a = py::none());

    // o3c.Tensor(((0, 1, 2), (3, 4, 5)), dtype=None, device=None).
    tensor.def(py::init([](const py::tuple& shape, std::optional<Dtype> dtype,
                           std::optional<Device> device) {
                   return PyTupleToTensor(shape, dtype, device);
               }),
               "Initialize Tensor from a nested tuple.", "shape"_a,
               "dtype"_a = py::none(), "device"_a = py::none());

    docstring::ClassMethodDocInject(m, "Tensor", "__init__", argument_docs);

    pybind_core_tensor_accessor(tensor);

    // Tensor creation API.
    BindTensorCreation(m, tensor, "empty", Tensor::Empty);
    BindTensorCreation(m, tensor, "zeros", Tensor::Zeros);
    BindTensorCreation(m, tensor, "ones", Tensor::Ones);
    BindTensorFullCreation<float>(m, tensor);
    BindTensorFullCreation<double>(m, tensor);
    BindTensorFullCreation<int8_t>(m, tensor);
    BindTensorFullCreation<int16_t>(m, tensor);
    BindTensorFullCreation<int32_t>(m, tensor);
    BindTensorFullCreation<int64_t>(m, tensor);
    BindTensorFullCreation<uint8_t>(m, tensor);
    BindTensorFullCreation<uint16_t>(m, tensor);
    BindTensorFullCreation<uint32_t>(m, tensor);
    BindTensorFullCreation<uint64_t>(m, tensor);
    BindTensorFullCreation<bool>(m, tensor);
    docstring::ClassMethodDocInject(m, "Tensor", "full", argument_docs);

    // Pickling support.
    // The tensor will be on the same device after deserialization.
    // Non contiguous tensors will be converted to contiguous tensors after
    // deserialization.
    tensor.def(py::pickle(
            [](const Tensor& t) {
                // __getstate__
                return py::make_tuple(t.GetDevice(),
                                      TensorToPyArray(t.To(Device("CPU:0"))));
            },
            [](py::tuple t) {
                // __setstate__
                if (t.size() != 2) {
                    utility::LogError(
                            "Cannot unpickle Tensor! Expecting a tuple of size "
                            "2.");
                }
                const Device& device = t[0].cast<Device>();
                if (!device.IsAvailable()) {
                    utility::LogWarning(
                            "Device {} is not available, tensor will be "
                            "created on CPU.",
                            device.ToString());
                    return PyArrayToTensor(t[1].cast<py::array>(), true);
                } else {
                    return PyArrayToTensor(t[1].cast<py::array>(), true)
                            .To(device);
                }
            }));

    tensor.def_static(
            "eye",
            [](int64_t n, std::optional<Dtype> dtype,
               std::optional<Device> device) {
                return Tensor::Eye(
                        n, dtype.has_value() ? dtype.value() : core::Float32,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "Create an identity matrix of size n x n.", "n"_a,
            "dtype"_a = py::none(), "device"_a = py::none());
    tensor.def_static("diag", &Tensor::Diag);

    // Tensor creation from arange for int.
    tensor.def_static(
            "arange",
            [](int64_t stop, std::optional<Dtype> dtype,
               std::optional<Device> device) {
                return Tensor::Arange(
                        0, stop, 1,
                        dtype.has_value() ? dtype.value() : core::Int64,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "Create a 1D tensor with evenly spaced values in the given "
            "interval.",
            "stop"_a, py::pos_only(), py::kw_only(), "dtype"_a = py::none(),
            "device"_a = py::none());
    tensor.def_static(
            "arange",
            [](int64_t start, int64_t stop, std::optional<int64_t> step,
               std::optional<Dtype> dtype, std::optional<Device> device) {
                return Tensor::Arange(
                        start, stop, step.has_value() ? step.value() : 1,
                        dtype.has_value() ? dtype.value() : core::Int64,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "Create a 1D tensor with evenly spaced values in the given "
            "interval.",
            "start"_a, "stop"_a, "step"_a = py::none(), "dtype"_a = py::none(),
            py::kw_only(), "device"_a = py::none());

    // Tensor creation from arange for float.
    tensor.def_static(
            "arange",
            [](double stop, std::optional<Dtype> dtype,
               std::optional<Device> device) {
                return Tensor::Arange(
                        0.0, stop, 1.0,
                        dtype.has_value() ? dtype.value() : core::Float64,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "Create a 1D tensor with evenly spaced values in the given "
            "interval.",
            "stop"_a, py::pos_only(), py::kw_only(), "dtype"_a = py::none(),
            "device"_a = py::none());
    tensor.def_static(
            "arange",
            [](double start, double stop, std::optional<double> step,
               std::optional<Dtype> dtype, std::optional<Device> device) {
                return Tensor::Arange(
                        start, stop, step.has_value() ? step.value() : 1.0,
                        dtype.has_value() ? dtype.value() : core::Float64,
                        device.has_value() ? device.value() : Device("CPU:0"));
            },
            "Create a 1D tensor with evenly spaced values in the given "
            "interval.",
            "start"_a, "stop"_a, "step"_a = py::none(), "dtype"_a = py::none(),
            py::kw_only(), "device"_a = py::none());

    tensor.def(
            "append",
            [](const Tensor& tensor, const Tensor& values,
               const std::optional<int64_t> axis) {
                if (axis.has_value()) {
                    return tensor.Append(values, axis);
                }
                return tensor.Append(values);
            },
            R"(Appends the `values` tensor, along the given axis and returns
a copy of the original tensor. Both the tensors must have same data-type
device, and number of dimensions. All dimensions must be the same, except the
dimension along the axis the tensors are to be appended.

This is the similar to NumPy's semantics:
- https://numpy.org/doc/stable/reference/generated/numpy.append.html

Returns:
    A copy of the tensor with `values` appended to axis. Note that append
    does not occur in-place: a new array is allocated and filled. If axis
    is None, out is a flattened tensor.

Example:
    >>> a = o3d.core.Tensor([[0, 1], [2, 3]])
    >>> b = o3d.core.Tensor([[4, 5]])
    >>> a.append(b, axis = 0)
    [[0 1],
     [2 3],
     [4 5]]
    Tensor[shape={3, 2}, stride={2, 1}, Int64, CPU:0, 0x55555abc6b00]

    >>> a.append(b)
    [0 1 2 3 4 5]
    Tensor[shape={6}, stride={1}, Int64, CPU:0, 0x55555abc6b70])",
            "values"_a, "axis"_a = py::none());

    // Device transfer.
    tensor.def(
            "cpu",
            [](const Tensor& tensor) {
                return tensor.To(core::Device("CPU:0"));
            },
            "Transfer the tensor to CPU. If the tensor "
            "is already on CPU, no copy will be performed.");
    tensor.def(
            "cuda",
            [](const Tensor& tensor, int device_id) {
                return tensor.To(core::Device("CUDA", device_id));
            },
            "Transfer the tensor to a CUDA device. If the tensor is already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);

    // Buffer I/O for Numpy and DLPack(PyTorch).
    tensor.def("numpy", &core::TensorToPyArray);

    tensor.def_static("from_numpy", [](py::array np_array) {
        return core::PyArrayToTensor(np_array, /*inplace=*/true);
    });

    auto to_dlpack = [](const Tensor& tensor,
                        std::optional<int> stream = std::nullopt,
                        std::optional<std::pair<int, int>> max_version =
                                std::nullopt,
                        std::optional<std::pair<DLDeviceType, int>> dl_device =
                                std::nullopt,
                        std::optional<bool> copy = std::nullopt) {
        bool versioned =
                (max_version.has_value() && max_version.value().first > 0);
        auto out_tensor = maybeCopyTensor(tensor, dl_device, copy);
        if (versioned) {
            auto capsule_destructor = [](PyObject* data) {
                DLManagedTensorVersioned* dl_managed_tensor =
                        (DLManagedTensorVersioned*)PyCapsule_GetPointer(
                                data, "dltensor");
                if (dl_managed_tensor != nullptr &&
                    dl_managed_tensor->deleter != nullptr) {
                    // the dl_managed_tensor has not been consumed,
                    // call deleter ourselves
                    dl_managed_tensor->deleter(
                            const_cast<DLManagedTensorVersioned*>(
                                    dl_managed_tensor));
                } else {
                    // The dl_managed_tensor has been consumed
                    // PyCapsule_GetPointer has set an error indicator
                    PyErr_Clear();
                }
            };
            DLManagedTensorVersioned* dlmt = out_tensor.ToDLPackVersioned();
            return py::capsule(dlmt, "dltensor", capsule_destructor);
        } else {
            auto capsule_destructor = [](PyObject* data) {
                DLManagedTensor* dl_managed_tensor =
                        (DLManagedTensor*)PyCapsule_GetPointer(data,
                                                               "dltensor");
                if (dl_managed_tensor != nullptr &&
                    dl_managed_tensor->deleter != nullptr) {
                    // the dl_managed_tensor has not been consumed,
                    // call deleter ourselves
                    dl_managed_tensor->deleter(
                            const_cast<DLManagedTensor*>(dl_managed_tensor));
                } else {
                    // The dl_managed_tensor has been consumed
                    // PyCapsule_GetPointer has set an error indicator
                    PyErr_Clear();
                }
            };
            DLManagedTensor* dlmt = out_tensor.ToDLPack();
            return py::capsule(dlmt, "dltensor", capsule_destructor);
        }
    };
    tensor.def(
            "__dlpack__", to_dlpack, "stream"_a = py::none(),
            "max_version"_a = py::none(), "dl_device"_a = py::none(),
            "copy"_a = py::none(),
            R"(Returns an opaque object (a "DLPack capsule") representing the tensor.

.. note::
  ``to_dlpack`` is a legacy DLPack interface. The capsule it returns
  cannot be used for anything in Python other than use it as input to
  ``from_dlpack``. The more idiomatic use of DLPack is to call
  ``from_dlpack`` directly on the tensor object - this works when that
  object has a ``__dlpack__`` method, which PyTorch and most other
  libraries indeed have now.

.. warning::
  Only call ``from_dlpack`` once per capsule produced with ``to_dlpack``.
  Behavior when a capsule is consumed multiple times is undefined.

Args:
    tensor: a tensor to be exported

The DLPack capsule shares the tensor's memory.)");
    tensor.attr("to_dlpack") = tensor.attr("__dlpack__");

    tensor.def("__dlpack_device__", [](const Tensor& tensor) {
        // &Open3DDLManagedTensor::getDLPackDevice
        // Prepare dl_device_type
        DLDeviceType dl_device_type;
        Device device = tensor.GetDevice();
        switch (device.GetType()) {
            case Device::DeviceType::CPU:
                dl_device_type = DLDeviceType::kDLCPU;
                break;
            case Device::DeviceType::CUDA:
                dl_device_type = DLDeviceType::kDLCUDA;
                break;
            case Device::DeviceType::SYCL:
                dl_device_type = DLDeviceType::kDLOneAPI;
                break;
            default:
                utility::LogError("ToDLPack: unsupported device type {}",
                                  device.ToString());
        }
        return std::make_pair(dl_device_type, device.GetID());
    });

    auto from_dlpack_capsule = [](py::capsule data) {
        auto invalid_capsule_err =
                "from_dlpack received an invalid capsule. Note that "
                "DLTensor capsules can be consumed only once, so you might "
                "have already constructed a tensor from it once.";
        Tensor t;
        // check versioned
        if (PyCapsule_IsValid(data, "dltensor_versioned")) {
            DLManagedTensorVersioned* dl_managed_tensor =
                    static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(
                            data.ptr(), "dltensor_versioned"));
            if (!dl_managed_tensor) {
                utility::LogError(invalid_capsule_err);
            }
            if (dl_managed_tensor->version.major > DLPACK_MAJOR_VERSION) {
                utility::LogError(
                        "Received DLPack capsule with major version {} > {} "
                        "(Max supported version).",
                        dl_managed_tensor->version.major, DLPACK_MAJOR_VERSION);
            }
            // Make sure that the PyCapsule is not used again. See:
            // torch/csrc/Module.cpp, and
            // https://github.com/cupy/cupy/pull/1445/files#diff-ddf01ff512087ef616db57ecab88c6ae
            t = Tensor::FromDLPackVersioned(dl_managed_tensor);
            PyCapsule_SetName(data.ptr(), "used_dltensor_versioned");
        } else {
            DLManagedTensor* dl_managed_tensor = static_cast<DLManagedTensor*>(
                    PyCapsule_GetPointer(data.ptr(), "dltensor"));
            if (!dl_managed_tensor) {
                utility::LogError(invalid_capsule_err);
            }
            t = Tensor::FromDLPack(dl_managed_tensor);
            PyCapsule_SetName(data.ptr(), "used_dltensor");
        }
        PyCapsule_SetDestructor(data.ptr(), nullptr);
        return t;
    };
    tensor.def_static("from_dlpack", from_dlpack_capsule, "dlpack_capsule"_a);
    tensor.def_static(
            "from_dlpack",
            [from_dlpack_capsule](
                    const py::object& ext_tensor,
                    std::optional<core::Device> device = std::nullopt,
                    std::optional<bool> copy = std::nullopt) {
                if (!hasattr(ext_tensor, "__dlpack__")) {
                    utility::LogError(
                            "from_dlpack: object does not define __dlpack__.");
                }
                py::object capsule_obj = ext_tensor.attr("__dlpack__")();
                auto t = from_dlpack_capsule(
                        py::reinterpret_borrow<py::capsule>(capsule_obj));
                return t;
            },
            "ext_tensor"_a, "device"_a = py::none(), "copy"_a = py::none(),
            R"(Converts a tensor from an external library into an Open3D ``Tensor``.

    The returned Open3D tensor will share the memory with the input tensor
    (which may have come from another library). Note that in-place operations
    will therefore also affect the data of the input tensor. This may lead to
    unexpected issues (e.g., other libraries may have read-only flags or
    immutable data structures), so the user should only do this if they know
    for sure that this is fine.

    Args:
        ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
            The tensor or DLPack capsule to convert.

            If ``ext_tensor`` is a tensor (or ndarray) object, it must support
            the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
            method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
            an opaque ``PyCapsule`` instance, typically produced by a
            ``to_dlpack`` function or method.

        device (open3d.core.device or None): An optional Open3D device
            specifying where to place the new tensor. Only ``None`` is supported
            (same device as ``ext_tensor``).

        copy (bool or None): An optional boolean indicating whether or not to copy
            ``self``. This is not supported yet and no copy is made.

    Examples::

        )");

    // Numpy IO.
    tensor.def("save", &Tensor::Save, "Save tensor to Numpy's npy format.",
               "file_name"_a);
    tensor.def_static("load", &Tensor::Load,
                      "Load tensor from Numpy's npy format.", "file_name"_a);

    /// Linalg operations.
    tensor.def("det", &Tensor::Det,
               "Compute the determinant of a 2D square tensor.");
    tensor.def(
            "lu_ipiv", &Tensor::LUIpiv,
            R"(Computes LU factorisation of the 2D square tensor, using A = P * L * U;
where P is the permutation matrix, L is the lower-triangular matrix with
diagonal elements as 1.0 and U is the upper-triangular matrix, and returns
tuple `output` tensor of shape {n,n} and `ipiv` tensor of shape {n}, where
{n,n} is the shape of input tensor.

Returns:
    ipiv: ipiv is a 1D integer pivot indices tensor. It contains the pivot
        indices, indicating row i of the matrix was interchanged with row
        ipiv[i]
    output: It has L as lower triangular values and U as upper triangle
        values including the main diagonal (diagonal elements of L to be
        taken as unity).

Example:
    >>> ipiv, output = a.lu_ipiv())");
    tensor.def("matmul", &Tensor::Matmul,
               "Computes matrix multiplication of a"
               " 2D tensor with another tensor of compatible shape.");
    tensor.def("__matmul__", &Tensor::Matmul,
               "Computes matrix multiplication of a"
               " 2D tensor with another tensor of compatible shape.");
    tensor.def("lstsq", &Tensor::LeastSquares,
               "Solves the linear system AX = B with QR decomposition and "
               "returns X. A is a (m, n) matrix with m >= n.",
               "B"_a);
    tensor.def("solve", &Tensor::Solve,
               "Solves the linear system AX = B with LU decomposition and "
               "returns X.  A must be a square matrix.",
               "B"_a);
    tensor.def("inv", &Tensor::Inverse,
               "Computes the matrix inverse of the square matrix self with "
               "LU factorization and returns the result.");
    tensor.def("svd", &Tensor::SVD,
               "Computes the matrix SVD decomposition :math:`A = U S V^T` and "
               "returns "
               "the result.  Note :math:`V^T` (V transpose) is returned "
               "instead of :math:`V`.");
    tensor.def("triu", &Tensor::Triu,
               "Returns the upper triangular matrix of the 2D tensor, above "
               "the given diagonal index. [The value of diagonal = col - row, "
               "therefore 0 is the main diagonal (row = col), and it shifts "
               "towards right for positive values (for diagonal = 1, col - "
               "row "
               "= 1), and towards left for negative values. The value of the "
               "diagonal parameter must be between [-m, n] for a {m,n} shaped "
               "tensor.",
               "diagonal"_a = 0);
    docstring::ClassMethodDocInject(m, "Tensor", "triu",
                                    {{"diagonal",
                                      "Value of [col - row], above which the "
                                      "elements are to be taken for"
                                      " upper triangular matrix."}});

    tensor.def("tril", &Tensor::Tril,
               "Returns the lower triangular matrix of the 2D tensor, above "
               "the given diagonal index. [The value of diagonal = col - row, "
               "therefore 0 is the main diagonal (row = col), and it shifts "
               "towards right for positive values (for diagonal = 1, col - "
               "row "
               "= 1), and towards left for negative values. The value of the "
               "diagonal parameter must be between [-m, n] where {m, n} is "
               "the "
               "shape of input tensor.",
               "diagonal"_a = 0);
    docstring::ClassMethodDocInject(m, "Tensor", "tril",
                                    {{"diagonal",
                                      "Value of [col - row], below which "
                                      "the elements are to be taken "
                                      "for lower triangular matrix."}});

    tensor.def("triul", &Tensor::Triul,
               "Returns the tuple of upper and lower triangular matrix of the "
               "2D "
               "tensor, above and below the given diagonal index.  The "
               "diagonal "
               "elements of lower triangular matrix are taken to be unity.  "
               "[The "
               "value of diagonal = col - row, therefore 0 is the main "
               "diagonal "
               "(row = col), and it shifts towards right for positive values "
               "(for "
               "diagonal = 1, col - row = 1), and towards left for negative "
               "values.  The value of the diagonal parameter must be between "
               "[-m, "
               "n] where {m, n} is the shape of input tensor.",
               "diagonal"_a = 0);
    docstring::ClassMethodDocInject(
            m, "Tensor", "triul",
            {{"diagonal",
              "Value of [col - row], above and below which the elements "
              "are to "
              "be taken for upper (diag. included) and lower triangular "
              "matrix."}});
    tensor.def(
            "lu",
            [](const Tensor& tensor, bool permute_l) {
                return tensor.LU(permute_l);
            },
            "permute_l"_a = false,
            R"(Computes LU factorisation of the 2D square tensor, using A = P * L * U;
where P is the permutation matrix, L is the lower-triangular matrix with
diagonal elements as 1.0 and U is the upper-triangular matrix, and returns
tuple (P, L, U).

Returns:
    Tuple (P, L, U).)");
    docstring::ClassMethodDocInject(
            m, "Tensor", "lu", {{"permute_l", "If True, returns L as P * L."}});

    // Casting and copying.
    tensor.def(
            "to",
            [](const Tensor& tensor, Dtype dtype, bool copy) {
                return tensor.To(dtype, copy);
            },
            "Returns a tensor with the specified ``dtype``.", "dtype"_a,
            "copy"_a = false);
    tensor.def(
            "to",
            [](const Tensor& tensor, const Device& device, bool copy) {
                return tensor.To(device, copy);
            },
            "Returns a tensor with the specified ``device``.", "device"_a,
            "copy"_a = false);
    tensor.def(
            "to",
            [](const Tensor& tensor, const Device& device, Dtype dtype,
               bool copy) { return tensor.To(device, dtype, copy); },
            "Returns a tensor with the specified ``device`` and ``dtype``."
            "device"_a,
            "dtype"_a, "copy"_a = false);
    docstring::ClassMethodDocInject(m, "Tensor", "to", argument_docs);

    tensor.def("clone", &Tensor::Clone, "Copy Tensor to the same device.");
    tensor.def("T", &Tensor::T,
               "Transpose <=2-D tensor by swapping dimension 0 and 1."
               "0-D and 1-D Tensor remains the same.");
    tensor.def(
            "reshape", &Tensor::Reshape,
            R"(Returns a tensor with the same data and number of elements as input, but
with the specified shape. When possible, the returned tensor will be a view of
input. Otherwise, it will be a copy.

Contiguous inputs and inputs with compatible strides can be reshaped
without copying, but you should not depend on the copying vs. viewing
behavior.

Ref:
- https://pytorch.org/docs/stable/tensors.html
- aten/src/ATen/native/TensorShape.cpp
- aten/src/ATen/TensorUtils.cpp)",
            "dst_shape"_a);
    docstring::ClassMethodDocInject(m, "Tensor", "reshape",
                                    {{"dst_shape",
                                      "Compatible destination shape with the "
                                      "same number of elements."}});
    tensor.def("contiguous", &Tensor::Contiguous,
               "Returns a contiguous tensor containing the same data in the "
               "same device.  If the tensor is already contiguous, the same "
               "underlying memory will be used.");
    tensor.def("is_contiguous", &Tensor::IsContiguous,
               "Returns True if the underlying memory buffer is contiguous.");
    tensor.def(
            "flatten", &Tensor::Flatten,
            R"(Flattens input by reshaping it into a one-dimensional tensor. If
start_dim or end_dim are passed, only dimensions starting with start_dim
and ending with end_dim are flattened. The order of elements in input is
unchanged.

Unlike NumPy’s flatten, which always copies input’s data, this function
may return the original object, a view, or copy. If no dimensions are
flattened, then the original object input is returned. Otherwise, if
input can be viewed as the flattened shape, then that view is returned.
Finally, only if the input cannot be viewed as the flattened shape is
input’s data copied.

Ref:
- https://pytorch.org/docs/stable/tensors.html
- aten/src/ATen/native/TensorShape.cpp
- aten/src/ATen/TensorUtils.cpp)",
            "start_dim"_a = 0, "end_dim"_a = -1);
    docstring::ClassMethodDocInject(
            m, "Tensor", "flatten",
            {{"start_dim", "The first dimension to flatten (inclusive)."},
             {"end_dim",
              "The last dimension to flatten, starting from start_dim "
              "(inclusive)."}});

    // See "emulating numeric types" section for Python built-in numeric
    // ops.
    // https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
    //
    // BinaryEW: add.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(add, Add, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(add_, Add_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__add__, Add, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__iadd__, Add_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__radd__, Add);

    // BinaryEW: sub.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(sub, Sub, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(sub_, Sub_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__sub__, Sub, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__isub__, Sub_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rsub__, Sub);

    // BinaryEW: mul.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(mul, Mul, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(mul_, Mul_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__mul__, Mul, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__imul__, Mul_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rmul__, Mul);

    // BinaryEW: div.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(div, Div, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(div_, Div_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__div__, Div, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__idiv__, Div_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rdiv__, Div);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__truediv__, Div, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__itruediv__, Div_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rtruediv__, Div);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__floordiv__, Div,
                                          CONST_ARG);  // truediv only.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__ifloordiv__, Div_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rfloordiv__, Div);

    // BinaryEW: and.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(logical_and, LogicalAnd, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(logical_and_, LogicalAnd_,
                                          NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__and__, LogicalAnd, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__iand__, LogicalAnd_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rand__, LogicalAnd);

    // BinaryEW: or.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(logical_or, LogicalOr, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(logical_or_, LogicalOr_,
                                          NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__or__, LogicalOr, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__ior__, LogicalOr_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__ror__, LogicalOr);

    // BinaryEW: xor.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(logical_xor, LogicalXor, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(logical_xor_, LogicalXor_,
                                          NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__xor__, LogicalXor, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__ixor__, LogicalXor_, NON_CONST_ARG);
    BIND_BINARY_R_OP_ALL_DTYPES(__rxor__, LogicalXor);

    // BinaryEW: comparison ops.
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(gt, Gt, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(gt_, Gt_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__gt__, Gt, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(lt, Lt, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(lt_, Lt_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__lt__, Lt, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(ge, Ge, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(ge_, Ge_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__ge__, Ge, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(le, Le, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(le_, Le_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__le__, Le, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(eq, Eq, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(eq_, Eq_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__eq__, Eq, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(ne, Ne, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(ne_, Ne_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES_WITH_SCALAR(__ne__, Ne, CONST_ARG);

    // Getters and setters as properties.
    tensor.def_property_readonly(
            "shape", [](const Tensor& tensor) { return tensor.GetShape(); });
    tensor.def_property_readonly("strides", [](const Tensor& tensor) {
        return tensor.GetStrides();
    });
    tensor.def_property_readonly("dtype", &Tensor::GetDtype);
    tensor.def_property_readonly("blob", &Tensor::GetBlob);
    tensor.def_property_readonly("ndim", &Tensor::NumDims);
    tensor.def("num_elements", &Tensor::NumElements);
    tensor.def("__bool__", &Tensor::IsNonZero);  // Python 3.X.

    tensor.def_property_readonly("device", &Tensor::GetDevice);
    tensor.def_property_readonly("is_cpu", &Tensor::IsCPU);
    tensor.def_property_readonly("is_cuda", &Tensor::IsCUDA);

    // Length and iterator.
    tensor.def("__len__", &Tensor::GetLength);
    tensor.def(
            "__iter__",
            [](Tensor& tensor) {
                return py::make_iterator(tensor.begin(), tensor.end());
            },
            py::keep_alive<0, 1>());  // Keep object alive while iterator
                                      // exists

    // Unary element-wise ops.
    tensor.def("sqrt", &Tensor::Sqrt);
    tensor.def("sqrt_", &Tensor::Sqrt_);
    tensor.def("sin", &Tensor::Sin);
    tensor.def("sin_", &Tensor::Sin_);
    tensor.def("cos", &Tensor::Cos);
    tensor.def("cos_", &Tensor::Cos_);
    tensor.def("neg", &Tensor::Neg);
    tensor.def("neg_", &Tensor::Neg_);
    tensor.def("__neg__", &Tensor::Neg);
    tensor.def("exp", &Tensor::Exp);
    tensor.def("exp_", &Tensor::Exp_);
    tensor.def("abs", &Tensor::Abs);
    tensor.def("abs_", &Tensor::Abs_);
    tensor.def("isnan", &Tensor::IsNan);
    tensor.def("isinf", &Tensor::IsInf);
    tensor.def("isfinite", &Tensor::IsFinite);
    tensor.def("floor", &Tensor::Floor);
    tensor.def("ceil", &Tensor::Ceil);
    tensor.def("round", &Tensor::Round);
    tensor.def("trunc", &Tensor::Trunc);
    tensor.def("logical_not", &Tensor::LogicalNot);
    tensor.def("logical_not_", &Tensor::LogicalNot_);

    BIND_CLIP_SCALAR(clip, Clip, CONST_ARG);
    BIND_CLIP_SCALAR(clip_, Clip_, NON_CONST_ARG);

    // Boolean.
    tensor.def(
            "nonzero",
            [](const Tensor& tensor, bool as_tuple) -> py::object {
                if (as_tuple) {
                    return py::cast(tensor.NonZero());
                } else {
                    return py::cast(tensor.NonZeroNumpy());
                }
            },
            "Find the indices of the elements that are non-zero.",
            "as_tuple"_a = false);
    docstring::ClassMethodDocInject(
            m, "Tensor", "nonzero",
            {{"as_tuple",
              "If ``as_tuple`` is True, returns an int64 tensor of shape "
              "{num_dims, num_non_zeros}, where the i-th row contains the "
              "indices of the non-zero elements in i-th dimension of the "
              "original tensor. If ``as_tuple`` is False, Returns a vector "
              "of "
              "int64 Tensors, each containing the indices of the non-zero "
              "elements in each dimension."}});
    tensor.def("all", &Tensor::All, py::call_guard<py::gil_scoped_release>(),
               py::arg("dim") = py::none(), py::arg("keepdim") = false,
               "Returns true if all elements in the tensor are true. Only "
               "works "
               "for boolean tensors.");
    tensor.def("any", &Tensor::Any, py::call_guard<py::gil_scoped_release>(),
               py::arg("dim") = py::none(), py::arg("keepdim") = false,
               "Returns true if any elements in the tensor are true. Only "
               "works "
               "for boolean tensors.");

    // Reduction ops.
    BIND_REDUCTION_OP(sum, Sum);
    BIND_REDUCTION_OP(mean, Mean);
    BIND_REDUCTION_OP(prod, Prod);
    BIND_REDUCTION_OP(min, Min);
    BIND_REDUCTION_OP(max, Max);
    BIND_REDUCTION_OP_NO_KEEPDIM(argmin, ArgMin);
    BIND_REDUCTION_OP_NO_KEEPDIM(argmax, ArgMax);

    // Comparison.
    tensor.def(
            "allclose", &Tensor::AllClose, "other"_a, "rtol"_a = 1e-5,
            "atol"_a = 1e-8,
            R"(Returns true if the two tensors are element-wise equal within a tolerance.

- If the ``device`` is not the same: throws exception.
- If the ``dtype`` is not the same: throws exception.
- If the ``shape`` is not the same: returns false.
- Returns true if: ``abs(self - other) <= (atol + rtol * abs(other)``).

The equation is not symmetrical, i.e. ``a.allclose(b)`` might not be the same
as ``b.allclose(a)``. Also see `Numpy's documentation <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>`__.

TODO:
	Support nan.)");
    docstring::ClassMethodDocInject(
            m, "Tensor", "allclose",
            {{"other", "The other tensor to compare with."},
             {"rtol", "Relative tolerance."},
             {"atol", "Absolute tolerance."}});
    tensor.def("isclose", &Tensor::IsClose, "other"_a, "rtol"_a = 1e-5,
               "atol"_a = 1e-8,
               R"(Element-wise version of ``tensor.allclose``.

- If the ``device`` is not the same: throws exception.
- If the ``dtype`` is not the same: throws exception.
- If the ``shape`` is not the same: throws exception.
- For each element in the returned tensor:
  ``abs(self - other) <= (atol + rtol * abs(other))``.

The equation is not symmetrical, i.e. a.is_close(b) might not be the same
as b.is_close(a). Also see `Numpy's documentation <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>`__.

TODO:
    Support nan.

Returns:
    A boolean tensor indicating where the tensor is close.)");
    docstring::ClassMethodDocInject(
            m, "Tensor", "isclose",
            {{"other", "The other tensor to compare with."},
             {"rtol", "Relative tolerance."},
             {"atol", "Absolute tolerance."}});

    tensor.def("issame", &Tensor::IsSame,
               "Returns true iff the tensor is the other tensor. This "
               "means that, the two tensors have the same underlying "
               "memory, device, dtype, shape, strides and etc.");
    // Print tensor.
    tensor.def("__repr__",
               [](const Tensor& tensor) { return tensor.ToString(); });
    tensor.def("__str__",
               [](const Tensor& tensor) { return tensor.ToString(); });

    // Get item from Tensor of one element.
    tensor.def(
            "item",
            [](const Tensor& tensor) -> py::object {
                Dtype dtype = tensor.GetDtype();
                if (dtype == core::Float32)
                    return py::float_(tensor.Item<float>());
                if (dtype == core::Float64)
                    return py::float_(tensor.Item<double>());
                if (dtype == core::Int8) return py::int_(tensor.Item<int8_t>());
                if (dtype == core::Int16)
                    return py::int_(tensor.Item<int16_t>());
                if (dtype == core::Int32)
                    return py::int_(tensor.Item<int32_t>());
                if (dtype == core::Int64)
                    return py::int_(tensor.Item<int64_t>());
                if (dtype == core::UInt8)
                    return py::int_(tensor.Item<uint8_t>());
                if (dtype == core::UInt16)
                    return py::int_(tensor.Item<uint16_t>());
                if (dtype == core::UInt32)
                    return py::int_(tensor.Item<uint32_t>());
                if (dtype == core::UInt64)
                    return py::int_(tensor.Item<uint64_t>());
                if (dtype == core::Bool) return py::bool_(tensor.Item<bool>());
                utility::LogError(
                        "Tensor.item(): unsupported dtype to convert to "
                        "python.");
                return py::none();
            },
            "Helper function to return the scalar value of a scalar "
            "tensor. "
            "The tensor must be 0 - dimensional (i.e. have an empty "
            "shape).");
}

}  // namespace core
}  // namespace open3d
