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

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashSet.h"
#include "open3d/utility/Logging.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/open3d_pybind.h"

namespace open3d {
namespace core {

const std::unordered_map<std::string, std::string> argument_docs = {
        {"init_capacity", "Initial capacity of a hash container."},
        {"key_dtype", "Data type for the input key tensor."},
        {"key_element_shape",
         "Element shape for the input key tensor. E.g. (3) for 3D "
         "coordinate keys."},
        {"value_dtype", "Data type for the input value tensor."},
        {"value_dtypes", "List of data type for the input value tensors."},
        {"value_element_shape",
         "Element shape for the input value tensor. E.g. (1) for mapped "
         "index."},
        {"value_element_shapes",
         "List of element shapes for the input value tensors. E.g. ((8,8,8,1), "
         "(8,8,8,3)) for "
         "mapped weights and RGB colors stored in 8^3 element arrays."},
        {"device",
         "Compute device to store and operate on the hash container."},
        {"copy",
         "If true, a new tensor is always created; if false, the copy is "
         "avoided when the original tensor already has the targeted dtype."},
        {"keys",
         "Input keys stored in a tensor of shape (N, key_element_shape)."},
        {"values",
         "Input values stored in a tensor of shape (N, value_element_shape)."},
        {"list_values",
         "List of input values stored in tensors of corresponding shapes."},
        {"capacity", "New capacity for rehashing."},
        {"file_name", "File name of the corresponding .npz file."},
        {"values_buffer_id", "Index of the value buffer tensor."},
        {"device_id", "Target CUDA device ID."}};

void pybind_core_hashmap(py::module& m) {
    py::class_<HashMap> hashmap(m, "HashMap",
                                "A HashMap is an unordered map from key to "
                                "value wrapped by Tensors.");

    hashmap.def(py::init<int64_t, const Dtype&, const SizeVector&, const Dtype&,
                         const SizeVector&, const Device&>(),
                "init_capacity"_a, "key_dtype"_a, "key_element_shape"_a,
                "value_dtype"_a, "value_element_shape"_a,
                "device"_a = Device("CPU:0"));
    hashmap.def(py::init<int64_t, const Dtype&, const SizeVector&,
                         const std::vector<Dtype>&,
                         const std::vector<SizeVector>&, const Device&>(),
                "init_capacity"_a, "key_dtype"_a, "key_element_shape"_a,
                "value_dtypes"_a, "value_element_shapes"_a,
                "device"_a = Device("CPU:0"));
    docstring::ClassMethodDocInject(m, "HashMap", "__init__", argument_docs);

    hashmap.def(
            "insert",
            [](HashMap& h, const Tensor& keys, const Tensor& values) {
                Tensor buf_indices, masks;
                h.Insert(keys, values, buf_indices, masks);
                return py::make_tuple(buf_indices, masks);
            },
            "Insert an array of keys and an array of values stored in Tensors.",
            "keys"_a, "values"_a);
    hashmap.def(
            "insert",
            [](HashMap& h, const Tensor& keys,
               const std::vector<Tensor>& values) {
                Tensor buf_indices, masks;
                h.Insert(keys, values, buf_indices, masks);
                return py::make_tuple(buf_indices, masks);
            },
            "Insert an array of keys and a list of value arrays stored in "
            "Tensors.",
            "keys"_a, "list_values"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "insert", argument_docs);

    hashmap.def(
            "activate",
            [](HashMap& h, const Tensor& keys) {
                Tensor buf_indices, masks;
                h.Activate(keys, buf_indices, masks);
                return py::make_tuple(buf_indices, masks);
            },
            "Activate an array of keys stored in Tensors without copying "
            "values.",
            "keys"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "activate", argument_docs);

    hashmap.def(
            "find",
            [](HashMap& h, const Tensor& keys) {
                Tensor buf_indices, masks;
                h.Find(keys, buf_indices, masks);
                return py::make_tuple(buf_indices, masks);
            },
            "Find an array of keys stored in Tensors.", "keys"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "find", argument_docs);

    hashmap.def(
            "erase",
            [](HashMap& h, const Tensor& keys) {
                Tensor masks;
                h.Erase(keys, masks);
                return masks;
            },
            "Erase an array of keys stored in Tensors.", "keys"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "erase", argument_docs);

    hashmap.def(
            "active_buf_indices",
            [](HashMap& h) {
                Tensor buf_indices;
                h.GetActiveIndices(buf_indices);
                return buf_indices;
            },
            "Get the buffer indices corresponding to active entries in the "
            "hash map.");

    hashmap.def("save", &HashMap::Save, "Save the hash map into a .npz file.",
                "file_name"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "save", argument_docs);

    hashmap.def_static("load", &HashMap::Load,
                       "Load a hash map from a .npz file.", "file_name"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "load", argument_docs);

    hashmap.def("reserve", &HashMap::Reserve,
                "Reserve the hash map given the capacity.", "capacity"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "reserve", argument_docs);

    hashmap.def("key_tensor", &HashMap::GetKeyTensor,
                "Get the key tensor stored in the buffer.");
    hashmap.def("value_tensors", &HashMap::GetValueTensors,
                "Get the list of value tensors stored in the buffer.");
    hashmap.def(
            "value_tensor", [](HashMap& h) { return h.GetValueTensor(); },
            "Get the value tensor stored at index 0.");
    hashmap.def(
            "value_tensor",
            [](HashMap& h, size_t i) { return h.GetValueTensor(i); },
            "Get the value tensor stored at index i", "value_buffer_id"_a);
    docstring::ClassMethodDocInject(m, "HashMap", "value_tensor",
                                    argument_docs);

    hashmap.def("size", &HashMap::Size, "Get the size of the hash map.");
    hashmap.def("capacity", &HashMap::GetCapacity,
                "Get the capacity of the hash map.");

    hashmap.def("clone", &HashMap::Clone,
                "Clone the hash map, including the data structure and the data "
                "buffers.");
    hashmap.def("to", &HashMap::To,
                "Convert the hash map to a selected device.", "device"_a,
                "copy"_a = false);
    docstring::ClassMethodDocInject(m, "HashMap", "to", argument_docs);

    hashmap.def(
            "cpu",
            [](const HashMap& hashmap) {
                return hashmap.To(core::Device("CPU:0"));
            },
            "Transfer the hash map to CPU. If the hash map "
            "is already on CPU, no copy will be performed.");
    hashmap.def(
            "cuda",
            [](const HashMap& hashmap, int device_id) {
                return hashmap.To(core::Device("CUDA", device_id));
            },
            "Transfer the hash map to a CUDA device. If the hash map is "
            "already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);
    docstring::ClassMethodDocInject(m, "HashMap", "cuda", argument_docs);

    hashmap.def_property_readonly("device", &HashMap::GetDevice);
    hashmap.def_property_readonly("is_cpu", &HashMap::IsCPU);
    hashmap.def_property_readonly("is_cuda", &HashMap::IsCUDA);
    hashmap.def_property_readonly("is_sycl", &HashMap::IsSYCL);
}

void pybind_core_hashset(py::module& m) {
    py::class_<HashSet> hashset(
            m, "HashSet",
            "A HashSet is an unordered set of keys wrapped by Tensors.");

    hashset.def(
            py::init<int64_t, const Dtype&, const SizeVector&, const Device&>(),
            "init_capacity"_a, "key_dtype"_a, "key_element_shape"_a,
            "device"_a = Device("CPU:0"));
    docstring::ClassMethodDocInject(m, "HashSet", "__init__", argument_docs);

    hashset.def(
            "insert",
            [](HashSet& h, const Tensor& keys) {
                Tensor buf_indices, masks;
                h.Insert(keys, buf_indices, masks);
                return py::make_tuple(buf_indices, masks);
            },
            "Insert an array of keys stored in Tensors.", "keys"_a);
    docstring::ClassMethodDocInject(m, "HashSet", "insert", argument_docs);

    hashset.def(
            "find",
            [](HashSet& h, const Tensor& keys) {
                Tensor buf_indices, masks;
                h.Find(keys, buf_indices, masks);
                return py::make_tuple(buf_indices, masks);
            },
            "Find an array of keys stored in Tensors.", "keys"_a);
    docstring::ClassMethodDocInject(m, "HashSet", "find", argument_docs);

    hashset.def(
            "erase",
            [](HashSet& h, const Tensor& keys) {
                Tensor masks;
                h.Erase(keys, masks);
                return masks;
            },
            "Erase an array of keys stored in Tensors.", "keys"_a);
    docstring::ClassMethodDocInject(m, "HashSet", "erase", argument_docs);

    hashset.def(
            "active_buf_indices",
            [](HashSet& h) {
                Tensor buf_indices;
                h.GetActiveIndices(buf_indices);
                return buf_indices;
            },
            "Get the buffer indices corresponding to active entries in the "
            "hash set.");

    hashset.def("save", &HashSet::Save, "Save the hash set into a .npz file.",
                "file_name"_a);
    docstring::ClassMethodDocInject(m, "HashSet", "save", argument_docs);

    hashset.def_static("load", &HashSet::Load,
                       "Load a hash set from a .npz file.", "file_name"_a);
    docstring::ClassMethodDocInject(m, "HashSet", "load", argument_docs);

    hashset.def("reserve", &HashSet::Reserve,
                "Reserve the hash set given the capacity.", "capacity"_a);
    docstring::ClassMethodDocInject(m, "HashSet", "reserve", argument_docs);

    hashset.def("key_tensor", &HashSet::GetKeyTensor,
                "Get the key tensor stored in the buffer.");

    hashset.def("size", &HashSet::Size, "Get the size of the hash set.");
    hashset.def("capacity", &HashSet::GetCapacity,
                "Get the capacity of the hash set.");

    hashset.def("clone", &HashSet::Clone,
                "Clone the hash set, including the data structure and the data "
                "buffers.");
    hashset.def("to", &HashSet::To,
                "Convert the hash set to a selected device.", "device"_a,
                "copy"_a = false);
    docstring::ClassMethodDocInject(m, "HashSet", "to", argument_docs);

    hashset.def(
            "cpu",
            [](const HashSet& hashset) {
                return hashset.To(core::Device("CPU:0"));
            },
            "Transfer the hash set to CPU. If the hash set "
            "is already on CPU, no copy will be performed.");
    hashset.def(
            "cuda",
            [](const HashSet& hashset, int device_id) {
                return hashset.To(core::Device("CUDA", device_id));
            },
            "Transfer the hash set to a CUDA device. If the hash set is "
            "already "
            "on the specified CUDA device, no copy will be performed.",
            "device_id"_a = 0);
    docstring::ClassMethodDocInject(m, "HashSet", "cuda", argument_docs);

    hashset.def_property_readonly("device", &HashSet::GetDevice);
    hashset.def_property_readonly("is_cpu", &HashSet::IsCPU);
    hashset.def_property_readonly("is_cuda", &HashSet::IsCUDA);
    hashset.def_property_readonly("is_sycl", &HashSet::IsSYCL);
}

}  // namespace core
}  // namespace open3d
