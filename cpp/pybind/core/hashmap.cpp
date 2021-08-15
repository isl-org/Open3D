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

#include "open3d/core/hashmap/Hashmap.h"

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/utility/Logging.h"
#include "pybind/core/core.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {
void pybind_core_hashmap(py::module& m) {
    py::class_<Hashmap> hashmap(
            m, "Hashmap",
            "A Hashmap is a map from key to data wrapped by Tensors.");

    hashmap.def(py::init([](int64_t init_capacity, const Dtype& dtype_key,
                            const py::handle& element_shape_key,
                            const Dtype& dtype_value,
                            const py::handle& element_shape_value,
                            const Device& device) {
                    SizeVector element_shape_key_sv =
                            PyHandleToSizeVector(element_shape_key);
                    SizeVector element_shape_value_sv =
                            PyHandleToSizeVector(element_shape_value);
                    return Hashmap(init_capacity, dtype_key,
                                   element_shape_key_sv, dtype_value,
                                   element_shape_value_sv, device);
                }),
                "init_capacity"_a, "dtype_key"_a, "element_shape_key"_a,
                "dtype_value"_a, "element_shape_value"_a,
                "device"_a = Device("CPU:0"));

    hashmap.def(
            py::init([](int64_t init_capacity, const Dtype& dtype_key,
                        const py::handle& element_shape_key,
                        const std::vector<Dtype>& dtypes_value,
                        const std::vector<py::handle>& element_shapes_value,
                        const Device& device) {
                SizeVector element_shape_key_sv =
                        PyHandleToSizeVector(element_shape_key);

                std::vector<SizeVector> element_shapes_value_sv;
                for (auto& handle : element_shapes_value) {
                    SizeVector element_shape_value_sv =
                            PyHandleToSizeVector(handle);
                    element_shapes_value_sv.push_back(element_shape_value_sv);
                }
                return Hashmap(init_capacity, dtype_key, element_shape_key_sv,
                               dtypes_value, element_shapes_value_sv, device);
            }),
            "init_capacity"_a, "dtype_key"_a, "element_shape_key"_a,
            "dtypes_value"_a, "element_shapes_value"_a,
            "device"_a = Device("CPU:0"));

    hashmap.def("insert",
                [](Hashmap& h, const Tensor& keys, const Tensor& values) {
                    Tensor buf_indices, masks;
                    h.Insert(keys, values, buf_indices, masks);
                    return py::make_tuple(buf_indices, masks);
                });
    hashmap.def("insert", [](Hashmap& h, const Tensor& keys,
                             const std::vector<Tensor>& values) {
        Tensor buf_indices, masks;
        h.Insert(keys, values, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashmap.def("activate", [](Hashmap& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Activate(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashmap.def("find", [](Hashmap& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Find(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashmap.def("erase", [](Hashmap& h, const Tensor& keys) {
        Tensor masks;
        h.Erase(keys, masks);
        return masks;
    });

    hashmap.def("get_active_buf_indices", [](Hashmap& h) {
        Tensor buf_indices;
        h.GetActiveIndices(buf_indices);
        return buf_indices;
    });

    hashmap.def("save", &Hashmap::Save);
    hashmap.def_static("load", &Hashmap::Load);

    hashmap.def("get_key_tensor", &Hashmap::GetKeyTensor);
    hashmap.def("get_value_tensors", &Hashmap::GetValueTensors);
    hashmap.def("get_value_tensor",
                [](Hashmap& h) { return h.GetValueTensor(); });
    hashmap.def("get_value_tensor",
                [](Hashmap& h, size_t i) { return h.GetValueTensor(i); });

    hashmap.def("rehash", &Hashmap::Rehash);
    hashmap.def("size", &Hashmap::Size);
    hashmap.def("capacity", &Hashmap::GetCapacity);

    hashmap.def("to", &Hashmap::To, "device"_a, "copy"_a = false);
    hashmap.def("clone", &Hashmap::Clone);
    hashmap.def(
            "cpu",
            [](const Hashmap& hashmap) {
                return hashmap.To(core::Device("CPU:0"));
            },
            "Transfer the hashmap to CPU. If the hashmap "
            "is already on CPU, no copy will be performed.");
    hashmap.def(
            "cuda",
            [](const Hashmap& hashmap, int device_id) {
                return hashmap.To(core::Device("CUDA", device_id));
            },
            "Transfer the hashmap to a CUDA device. If the hashmap is already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);

    py::class_<Hashset, Hashmap> hashset(m, "Hashset",
                                         "A Hashset is a unordered set that "
                                         "collects keys wrapped by Tensors.");
    hashset.def(py::init([](int64_t init_capacity, const Dtype& dtype_key,
                            const py::handle& element_shape_key,
                            const Device& device) {
                    SizeVector element_shape_key_sv =
                            PyHandleToSizeVector(element_shape_key);
                    return Hashset(init_capacity, dtype_key,
                                   element_shape_key_sv, device);
                }),
                "init_capacity"_a, "dtype_key"_a, "element_shape_key"_a,
                "device"_a = Device("CPU:0"));

    hashset.def("insert", [](Hashset& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Insert(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });
}

}  // namespace core
}  // namespace open3d
