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

#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/TensorKey.h"

#define CONST_ARG const
#define NON_CONST_ARG
#define BIND_BINARY_OP_ALL_DTYPES(py_name, cpp_name, self_const)              \
    tensor.def(#py_name,                                                      \
               [](self_const core::Tensor& self, const core::Tensor& other) { \
                   return self.cpp_name(other);                               \
               });                                                            \
    tensor.def(#py_name, &core::Tensor::cpp_name<float>);                     \
    tensor.def(#py_name, &core::Tensor::cpp_name<double>);                    \
    tensor.def(#py_name, &core::Tensor::cpp_name<int32_t>);                   \
    tensor.def(#py_name, &core::Tensor::cpp_name<int64_t>);                   \
    tensor.def(#py_name, &core::Tensor::cpp_name<uint8_t>);                   \
    tensor.def(#py_name, &core::Tensor::cpp_name<bool>);

namespace open3d {

template <typename T>
void bind_templated_constructor(py::class_<core::Tensor>& tensor) {
    tensor.def(
            py::init([](const std::vector<T>& init_vals,
                        const core::SizeVector& shape, const core::Dtype& dtype,
                        const core::Device& device = core::Device("CPU:0")) {
                return new core::Tensor(init_vals, shape, dtype, device);
            }),
            "init_vals"_a, "shape"_a, "dtype"_a, "device"_a);
}

template <typename T>
static std::vector<T> ToFlatVector(
        py::array_t<T, py::array::c_style | py::array::forcecast> np_array) {
    py::buffer_info info = np_array.request();
    T* start = static_cast<T*>(info.ptr);
    return std::vector<T>(start, start + info.size);
}

void pybind_core_tensor(py::module& m) {
    py::class_<core::Tensor, std::shared_ptr<core::Tensor>> tensor(
            m, "Tensor",
            "A Tensor is a view of a data Blob with shape, stride, data_ptr.");

    // Constructor from numpy array
    tensor.def(py::init([](py::array np_array, const core::Dtype& dtype,
                           const core::Device& device) {
        py::buffer_info info = np_array.request();
        core::SizeVector shape(info.shape.begin(), info.shape.end());
        core::Tensor t;
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            t = core::Tensor(ToFlatVector<scalar_t>(np_array), shape, dtype,
                             device);
        });
        return t;
    }));

    // Tensor creation API
    tensor.def_static("empty", &core::Tensor::Empty);
    tensor.def_static("full", &core::Tensor::Full<float>);
    tensor.def_static("full", &core::Tensor::Full<double>);
    tensor.def_static("full", &core::Tensor::Full<int32_t>);
    tensor.def_static("full", &core::Tensor::Full<int64_t>);
    tensor.def_static("full", &core::Tensor::Full<uint8_t>);
    tensor.def_static("full", &core::Tensor::Full<bool>);
    tensor.def_static("zeros", &core::Tensor::Zeros);
    tensor.def_static("ones", &core::Tensor::Ones);

    // Tensor copy
    tensor.def("shallow_copy_from", &core::Tensor::ShallowCopyFrom);

    // Device transfer
    tensor.def("cuda",
               [](const core::Tensor& tensor, int device_id = 0) {
                   if (!core::cuda::IsAvailable()) {
                       utility::LogError(
                               "CUDA is not available, cannot copy Tensor.");
                   }
                   if (device_id < 0 ||
                       device_id >= core::cuda::DeviceCount()) {
                       utility::LogError(
                               "Invalid device_id {}, must satisfy 0 <= "
                               "device_id < {}",
                               device_id, core::cuda::DeviceCount());
                   }
                   return tensor.Copy(core::Device(
                           core::Device::DeviceType::CUDA, device_id));
               })
            .def("cpu", [](const core::Tensor& tensor) {
                return tensor.Copy(
                        core::Device(core::Device::DeviceType::CPU, 0));
            });

    // Buffer I/O for Numpy and DLPack(PyTorch)
    tensor.def("numpy", [](const core::Tensor& tensor) {
        if (tensor.GetDevice().GetType() != core::Device::DeviceType::CPU) {
            utility::LogError(
                    "Can only convert CPU Tensor to numpy. Copy "
                    "Tensor to CPU before converting to numpy.");
        }
        py::dtype py_dtype =
                py::dtype(pybind_utils::DtypeToArrayFormat(tensor.GetDtype()));
        py::array::ShapeContainer py_shape(tensor.GetShape());
        core::SizeVector strides = tensor.GetStrides();
        int64_t element_byte_size =
                core::DtypeUtil::ByteSize(tensor.GetDtype());
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
        core::Tensor* base_tensor = new core::Tensor(tensor);

        // See PyTorch's torch/csrc/Module.cpp
        auto capsule_destructor = [](PyObject* data) {
            core::Tensor* base_tensor = reinterpret_cast<core::Tensor*>(
                    PyCapsule_GetPointer(data, "open3d::core::Tensor"));
            if (base_tensor) {
                delete base_tensor;
            } else {
                PyErr_Clear();
            }
        };

        py::capsule base_tensor_capsule(base_tensor, "open3d::core::Tensor",
                                        capsule_destructor);

        return py::array(py_dtype, py_shape, py_strides, tensor.GetDataPtr(),
                         base_tensor_capsule);
    });

    tensor.def_static("from_numpy", [](py::array np_array) {
        py::buffer_info info = np_array.request();

        core::SizeVector shape(info.shape.begin(), info.shape.end());
        core::SizeVector strides(info.strides.begin(), info.strides.end());
        for (size_t i = 0; i < strides.size(); ++i) {
            strides[i] /= info.itemsize;
        }
        core::Dtype dtype = pybind_utils::ArrayFormatToDtype(info.format);
        core::Device device("CPU:0");

        // Blob expects an std::function<void(void*)> deleter, a
        // dummy deleter is used here, since the memory is
        // managed by numpy.
        std::function<void(void*)> deleter = [](void*) -> void {};
        auto blob = std::make_shared<core::Blob>(device, info.ptr, deleter);

        return core::Tensor(shape, strides, info.ptr, dtype, blob);
    });

    tensor.def("to_dlpack", [](const core::Tensor& tensor) {
        DLManagedTensor* dl_managed_tensor = tensor.ToDLPack();
        // See PyTorch's torch/csrc/Module.cpp
        auto capsule_destructor = [](PyObject* data) {
            DLManagedTensor* dl_managed_tensor =
                    (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
            if (dl_managed_tensor) {
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
        return py::capsule(dl_managed_tensor, "dltensor", capsule_destructor);
    });

    tensor.def_static("from_dlpack", [](py::capsule data) {
        DLManagedTensor* dl_managed_tensor =
                static_cast<DLManagedTensor*>(data);
        if (!dl_managed_tensor) {
            utility::LogError(
                    "from_dlpack must receive "
                    "DLManagedTensor PyCapsule.");
        }
        // Make sure that the PyCapsule is not used again.
        // See:
        // torch/csrc/Module.cpp, and
        // https://github.com/cupy/cupy/pull/1445/files#diff-ddf01ff512087ef616db57ecab88c6ae
        core::Tensor t = core::Tensor::FromDLPack(dl_managed_tensor);
        PyCapsule_SetName(data.ptr(), "used_dltensor");
        return t;
    });

    tensor.def("_getitem",
               [](const core::Tensor& tensor, const core::TensorKey& tk) {
                   return tensor.GetItem(tk);
               });

    tensor.def("_getitem_vector", [](const core::Tensor& tensor,
                                     const std::vector<core::TensorKey>& tks) {
        return tensor.GetItem(tks);
    });

    tensor.def("_setitem", [](core::Tensor& tensor, const core::TensorKey& tk,
                              const core::Tensor& value) {
        return tensor.SetItem(tk, value);
    });

    tensor.def("_setitem_vector",
               [](core::Tensor& tensor, const std::vector<core::TensorKey>& tks,
                  const core::Tensor& value) {
                   return tensor.SetItem(tks, value);
               });

    // Casting
    tensor.def("to", &core::Tensor::To);

    // Binary element-wise ops
    BIND_BINARY_OP_ALL_DTYPES(add, Add, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(add_, Add_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(sub, Sub, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(sub_, Sub_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(mul, Mul, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(mul_, Mul_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(div, Div, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(div_, Div_, NON_CONST_ARG);

    // Binary boolean element-wise ops
    BIND_BINARY_OP_ALL_DTYPES(logical_and, LogicalAnd, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(logical_and_, LogicalAnd_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(logical_or, LogicalOr, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(logical_or_, LogicalOr_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(logical_xor, LogicalXor, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(logical_xor_, LogicalXor_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(gt, Gt, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(gt_, Gt_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(lt, Lt, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(lt_, Lt_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(ge, Ge, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(ge_, Ge_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(le, Le, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(le_, Le_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(eq, Eq, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(eq_, Eq_, NON_CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(ne, Ne, CONST_ARG);
    BIND_BINARY_OP_ALL_DTYPES(ne_, Ne_, NON_CONST_ARG);

    // Getters and setters as peoperty
    tensor.def_property_readonly("shape", [](const core::Tensor& tensor) {
        return tensor.GetShape();
    });
    tensor.def_property_readonly("strides", [](const core::Tensor& tensor) {
        return tensor.GetStrides();
    });
    tensor.def_property_readonly("dtype", &core::Tensor::GetDtype);
    tensor.def_property_readonly("device", &core::Tensor::GetDevice);
    tensor.def_property_readonly("blob", &core::Tensor::GetBlob);
    tensor.def_property_readonly("ndim", &core::Tensor::NumDims);
    tensor.def("num_elements", &core::Tensor::NumElements);

    // Unary element-wise ops
    tensor.def("sqrt", &core::Tensor::Sqrt);
    tensor.def("sqrt_", &core::Tensor::Sqrt_);
    tensor.def("sin", &core::Tensor::Sin);
    tensor.def("sin_", &core::Tensor::Sin_);
    tensor.def("cos", &core::Tensor::Cos);
    tensor.def("cos_", &core::Tensor::Cos_);
    tensor.def("neg", &core::Tensor::Neg);
    tensor.def("neg_", &core::Tensor::Neg_);
    tensor.def("exp", &core::Tensor::Exp);
    tensor.def("exp_", &core::Tensor::Exp_);
    tensor.def("abs", &core::Tensor::Abs);
    tensor.def("abs_", &core::Tensor::Abs_);
    tensor.def("logical_not", &core::Tensor::LogicalNot);
    tensor.def("logical_not_", &core::Tensor::LogicalNot_);

    // Boolean
    tensor.def("_non_zero", &core::Tensor::NonZero);
    tensor.def("_non_zero_numpy", &core::Tensor::NonZeroNumpy);
    tensor.def("all", &core::Tensor::All);
    tensor.def("any", &core::Tensor::Any);

    // Reduction ops
    tensor.def("sum", &core::Tensor::Sum);
    tensor.def("mean", &core::Tensor::Mean);
    tensor.def("prod", &core::Tensor::Prod);
    tensor.def("min", &core::Tensor::Min);
    tensor.def("max", &core::Tensor::Max);
    tensor.def("argmin_", &core::Tensor::ArgMin);
    tensor.def("argmax_", &core::Tensor::ArgMax);

    // Comparison
    tensor.def("allclose", &core::Tensor::AllClose, "other"_a, "rtol"_a = 1e-5,
               "atol"_a = 1e-8);
    tensor.def("isclose", &core::Tensor::IsClose, "other"_a, "rtol"_a = 1e-5,
               "atol"_a = 1e-8);
    tensor.def("issame", &core::Tensor::IsSame);

    // Print tensor
    tensor.def("__repr__",
               [](const core::Tensor& tensor) { return tensor.ToString(); });
    tensor.def("__str__",
               [](const core::Tensor& tensor) { return tensor.ToString(); });

    // Get item from Tensor of one element
    tensor.def("_item_float",
               [](const core::Tensor& t) { return t.Item<float>(); });
    tensor.def("_item_double",
               [](const core::Tensor& t) { return t.Item<double>(); });
    tensor.def("_item_int32_t",
               [](const core::Tensor& t) { return t.Item<int32_t>(); });
    tensor.def("_item_int64_t",
               [](const core::Tensor& t) { return t.Item<int64_t>(); });
    tensor.def("_item_uint8_t",
               [](const core::Tensor& t) { return t.Item<uint8_t>(); });
    tensor.def("_item_bool",
               [](const core::Tensor& t) { return t.Item<bool>(); });
}

}  // namespace open3d
