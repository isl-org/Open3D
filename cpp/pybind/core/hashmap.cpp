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

#include "open3d/core/hashmap/HashMap.h"

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashSet.h"
#include "open3d/utility/Logging.h"
#include "pybind/core/core.h"
#include "pybind/core/tensor_converter.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {
void pybind_core_hashmap(py::module& m) {
    py::class_<HashMap> hashmap(
            m, "HashMap",
            "A HashMap is a map from key to data wrapped by Tensors.");

    hashmap.def(py::init([](int64_t init_capacity, const Dtype& key_dtype,
                            const py::handle& key_element_shape,
                            const Dtype& value_dtype,
                            const py::handle& value_element_shape,
                            const Device& device) {
                    SizeVector key_element_shape_sv =
                            PyHandleToSizeVector(key_element_shape);
                    SizeVector value_element_shape_sv =
                            PyHandleToSizeVector(value_element_shape);
                    return HashMap(init_capacity, key_dtype,
                                   key_element_shape_sv, value_dtype,
                                   value_element_shape_sv, device);
                }),
                "init_capacity"_a, "key_dtype"_a, "key_element_shape"_a,
                "value_dtype"_a, "value_element_shape"_a,
                "device"_a = Device("CPU:0"));

    hashmap.def(
            py::init([](int64_t init_capacity, const Dtype& key_dtype,
                        const py::handle& key_element_shape,
                        const std::vector<Dtype>& dtypes_value,
                        const std::vector<py::handle>& element_shapes_value,
                        const Device& device) {
                SizeVector key_element_shape_sv =
                        PyHandleToSizeVector(key_element_shape);

                std::vector<SizeVector> element_shapes_value_sv;
                for (auto& handle : element_shapes_value) {
                    SizeVector value_element_shape_sv =
                            PyHandleToSizeVector(handle);
                    element_shapes_value_sv.push_back(value_element_shape_sv);
                }
                return HashMap(init_capacity, key_dtype, key_element_shape_sv,
                               dtypes_value, element_shapes_value_sv, device);
            }),
            "init_capacity"_a, "key_dtype"_a, "key_element_shape"_a,
            "dtypes_value"_a, "element_shapes_value"_a,
            "device"_a = Device("CPU:0"));

    hashmap.def("insert",
                [](HashMap& h, const Tensor& keys, const Tensor& values) {
                    Tensor buf_indices, masks;
                    h.Insert(keys, values, buf_indices, masks);
                    return py::make_tuple(buf_indices, masks);
                });
    hashmap.def("insert", [](HashMap& h, const Tensor& keys,
                             const std::vector<Tensor>& values) {
        Tensor buf_indices, masks;
        h.Insert(keys, values, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashmap.def("activate", [](HashMap& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Activate(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashmap.def("find", [](HashMap& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Find(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashmap.def("erase", [](HashMap& h, const Tensor& keys) {
        Tensor masks;
        h.Erase(keys, masks);
        return masks;
    });

    hashmap.def("get_active_buf_indices", [](HashMap& h) {
        Tensor buf_indices;
        h.GetActiveIndices(buf_indices);
        return buf_indices;
    });

    hashmap.def("save", &HashMap::Save);
    hashmap.def_static("load", &HashMap::Load);

    hashmap.def("get_key_tensor", &HashMap::GetKeyTensor);
    hashmap.def("get_value_tensors", &HashMap::GetValueTensors);
    hashmap.def("get_value_tensor",
                [](HashMap& h) { return h.GetValueTensor(); });
    hashmap.def("get_value_tensor",
                [](HashMap& h, size_t i) { return h.GetValueTensor(i); });

    hashmap.def("rehash", &HashMap::Rehash);
    hashmap.def("size", &HashMap::Size);
    hashmap.def("capacity", &HashMap::GetCapacity);

    hashmap.def("to", &HashMap::To, "device"_a, "copy"_a = false);
    hashmap.def("clone", &HashMap::Clone);
    hashmap.def(
            "cpu",
            [](const HashMap& hashmap) {
                return hashmap.To(core::Device("CPU:0"));
            },
            "Transfer the hashmap to CPU. If the hashmap "
            "is already on CPU, no copy will be performed.");
    hashmap.def(
            "cuda",
            [](const HashMap& hashmap, int device_id) {
                return hashmap.To(core::Device("CUDA", device_id));
            },
            "Transfer the hashmap to a CUDA device. If the hashmap is already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);

    /////////////////////////////////////////////////
    py::class_<HashSet> hashset(m, "HashSet",
                                "A HashSet is a unordered set that "
                                "collects keys wrapped by Tensors.");
    hashset.def(py::init([](int64_t init_capacity, const Dtype& key_dtype,
                            const py::handle& key_element_shape,
                            const Device& device) {
                    SizeVector key_element_shape_sv =
                            PyHandleToSizeVector(key_element_shape);
                    return HashSet(init_capacity, key_dtype,
                                   key_element_shape_sv, device);
                }),
                "init_capacity"_a, "key_dtype"_a, "key_element_shape"_a,
                "device"_a = Device("CPU:0"));

    hashset.def("insert", [](HashSet& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Insert(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashset.def("find", [](HashSet& h, const Tensor& keys) {
        Tensor buf_indices, masks;
        h.Find(keys, buf_indices, masks);
        return py::make_tuple(buf_indices, masks);
    });

    hashset.def("erase", [](HashSet& h, const Tensor& keys) {
        Tensor masks;
        h.Erase(keys, masks);
        return masks;
    });

    hashset.def("get_active_buf_indices", [](HashSet& h) {
        Tensor buf_indices;
        h.GetActiveIndices(buf_indices);
        return buf_indices;
    });

    hashset.def("save", &HashSet::Save);
    hashset.def_static("load", &HashSet::Load);

    hashset.def("get_key_tensor", &HashSet::GetKeyTensor);

    hashset.def("rehash", &HashSet::Rehash);
    hashset.def("size", &HashSet::Size);
    hashset.def("capacity", &HashSet::GetCapacity);

    hashset.def("to", &HashSet::To, "device"_a, "copy"_a = false);
    hashset.def("clone", &HashSet::Clone);
    hashset.def(
            "cpu",
            [](const HashSet& hashset) {
                return hashset.To(core::Device("CPU:0"));
            },
            "Transfer the hashset to CPU. If the hashset "
            "is already on CPU, no copy will be performed.");
    hashset.def(
            "cuda",
            [](const HashSet& hashset, int device_id) {
                return hashset.To(core::Device("CUDA", device_id));
            },
            "Transfer the hashset to a CUDA device. If the hashset is already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);
}

}  // namespace core
}  // namespace open3d
