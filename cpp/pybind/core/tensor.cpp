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

#include "open3d/core/Tensor.h"

#include <vector>

#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Device.h"
#include "open3d/core/Dispatch.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/SizeVector.h"
#include "open3d/core/TensorKey.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"
#include "pybind/pybind_utils.h"

#define CONST_ARG const
#define NON_CONST_ARG
#define BIND_BINARY_OP_ALL_DTYPES(py_name, cpp_name, self_const)            \
    tensor.def(#py_name, [](self_const Tensor& self, const Tensor& other) { \
        return self.cpp_name(other);                                        \
    });                                                                     \
    tensor.def(#py_name, &Tensor::cpp_name<float>);                         \
    tensor.def(#py_name, &Tensor::cpp_name<double>);                        \
    tensor.def(#py_name, &Tensor::cpp_name<int32_t>);                       \
    tensor.def(#py_name, &Tensor::cpp_name<int64_t>);                       \
    tensor.def(#py_name, &Tensor::cpp_name<uint8_t>);                       \
    tensor.def(#py_name, &Tensor::cpp_name<bool>);

namespace open3d {
namespace core {

template <typename T>
void bind_templated_constructor(py::class_<Tensor>& tensor) {
    tensor.def(py::init([](const std::vector<T>& init_vals,
                           const SizeVector& shape, const Dtype& dtype,
                           const Device& device = Device("CPU:0")) {
                   return new Tensor(init_vals, shape, dtype, device);
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
    py::class_<Tensor> tensor(
            m, "Tensor",
            "A Tensor is a view of a data Blob with shape, stride, data_ptr.");

    // Constructor from numpy array
    tensor.def(py::init([](py::array np_array, const Dtype& dtype,
                           const Device& device) {
        py::buffer_info info = np_array.request();
        SizeVector shape(info.shape.begin(), info.shape.end());
        Tensor t;
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dtype, [&]() {
            t = Tensor(ToFlatVector<scalar_t>(np_array), shape, dtype, device);
        });
        return t;
    }));

    // Tensor creation API
    tensor.def_static("empty", &Tensor::Empty);
    tensor.def_static("full", &Tensor::Full<float>);
    tensor.def_static("full", &Tensor::Full<double>);
    tensor.def_static("full", &Tensor::Full<int32_t>);
    tensor.def_static("full", &Tensor::Full<int64_t>);
    tensor.def_static("full", &Tensor::Full<uint8_t>);
    tensor.def_static("full", &Tensor::Full<bool>);
    tensor.def_static("zeros", &Tensor::Zeros);
    tensor.def_static("ones", &Tensor::Ones);
    tensor.def_static("eye", &Tensor::Eye);
    tensor.def_static("diag", &Tensor::Diag);

    // Tensor copy
    tensor.def("shallow_copy_from", &Tensor::ShallowCopyFrom);

    // Device transfer
    tensor.def("cuda", [](const Tensor& tensor, int device_id = 0) {
              if (!cuda::IsAvailable()) {
                  utility::LogError(
                          "CUDA is not available, cannot copy Tensor.");
              }
              if (device_id < 0 || device_id >= cuda::DeviceCount()) {
                  utility::LogError(
                          "Invalid device_id {}, must satisfy 0 <= "
                          "device_id < {}",
                          device_id, cuda::DeviceCount());
              }
              return tensor.Copy(Device(Device::DeviceType::CUDA, device_id));
          }).def("cpu", [](const Tensor& tensor) {
        return tensor.Copy(Device(Device::DeviceType::CPU, 0));
    });

    // Buffer I/O for Numpy and DLPack(PyTorch)
    tensor.def("numpy", &core::TensorToPyArray);

    tensor.def_static("from_numpy", [](py::array np_array) {
        return core::PyArrayToTensor(np_array, true);
    });

    tensor.def("to_dlpack", [](const Tensor& tensor) {
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
        Tensor t = Tensor::FromDLPack(dl_managed_tensor);
        PyCapsule_SetName(data.ptr(), "used_dltensor");
        return t;
    });

    /// Linalg operations
    tensor.def("matmul", &Tensor::Matmul);
    tensor.def("lstsq", &Tensor::LeastSquares);
    tensor.def("solve", &Tensor::Solve);
    tensor.def("inv", &Tensor::Inverse);
    tensor.def("svd", &Tensor::SVD);

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

    // Casting
    tensor.def("to", &Tensor::To);
    tensor.def("T", &Tensor::T);
    tensor.def("contiguous", &Tensor::Contiguous);

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
    tensor.def_property_readonly(
            "shape", [](const Tensor& tensor) { return tensor.GetShape(); });
    tensor.def_property_readonly("strides", [](const Tensor& tensor) {
        return tensor.GetStrides();
    });
    tensor.def_property_readonly("dtype", &Tensor::GetDtype);
    tensor.def_property_readonly("device", &Tensor::GetDevice);
    tensor.def_property_readonly("blob", &Tensor::GetBlob);
    tensor.def_property_readonly("ndim", &Tensor::NumDims);
    tensor.def("num_elements", &Tensor::NumElements);

    // Unary element-wise ops
    tensor.def("sqrt", &Tensor::Sqrt);
    tensor.def("sqrt_", &Tensor::Sqrt_);
    tensor.def("sin", &Tensor::Sin);
    tensor.def("sin_", &Tensor::Sin_);
    tensor.def("cos", &Tensor::Cos);
    tensor.def("cos_", &Tensor::Cos_);
    tensor.def("neg", &Tensor::Neg);
    tensor.def("neg_", &Tensor::Neg_);
    tensor.def("exp", &Tensor::Exp);
    tensor.def("exp_", &Tensor::Exp_);
    tensor.def("abs", &Tensor::Abs);
    tensor.def("abs_", &Tensor::Abs_);
    tensor.def("logical_not", &Tensor::LogicalNot);
    tensor.def("logical_not_", &Tensor::LogicalNot_);

    // Boolean
    tensor.def("_non_zero", &Tensor::NonZero);
    tensor.def("_non_zero_numpy", &Tensor::NonZeroNumpy);
    tensor.def("all", &Tensor::All);
    tensor.def("any", &Tensor::Any);

    // Reduction ops
    tensor.def("sum", &Tensor::Sum);
    tensor.def("mean", &Tensor::Mean);
    tensor.def("prod", &Tensor::Prod);
    tensor.def("min", &Tensor::Min);
    tensor.def("max", &Tensor::Max);
    tensor.def("argmin_", &Tensor::ArgMin);
    tensor.def("argmax_", &Tensor::ArgMax);

    // Comparison
    tensor.def("allclose", &Tensor::AllClose, "other"_a, "rtol"_a = 1e-5,
               "atol"_a = 1e-8);
    tensor.def("isclose", &Tensor::IsClose, "other"_a, "rtol"_a = 1e-5,
               "atol"_a = 1e-8);
    tensor.def("issame", &Tensor::IsSame);

    // Print tensor
    tensor.def("__repr__",
               [](const Tensor& tensor) { return tensor.ToString(); });
    tensor.def("__str__",
               [](const Tensor& tensor) { return tensor.ToString(); });

    // Get item from Tensor of one element
    tensor.def("_item_float", [](const Tensor& t) { return t.Item<float>(); });
    tensor.def("_item_double",
               [](const Tensor& t) { return t.Item<double>(); });
    tensor.def("_item_int32_t",
               [](const Tensor& t) { return t.Item<int32_t>(); });
    tensor.def("_item_int64_t",
               [](const Tensor& t) { return t.Item<int64_t>(); });
    tensor.def("_item_uint8_t",
               [](const Tensor& t) { return t.Item<uint8_t>(); });
    tensor.def("_item_bool", [](const Tensor& t) { return t.Item<bool>(); });
}

}  // namespace core
}  // namespace open3d
