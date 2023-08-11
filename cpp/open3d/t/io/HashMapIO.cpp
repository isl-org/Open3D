// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/io/HashMapIO.h"

#include "open3d/t/io/NumpyIO.h"
#include "open3d/utility/FileSystem.h"
namespace open3d {
namespace t {
namespace io {

void WriteHashMap(const std::string& file_name, const core::HashMap& hashmap) {
    core::Tensor keys = hashmap.GetKeyTensor();
    std::vector<core::Tensor> values = hashmap.GetValueTensors();

    core::Device host("CPU:0");

    core::Tensor active_buf_indices_i32;
    hashmap.GetActiveIndices(active_buf_indices_i32);
    core::Tensor active_indices = active_buf_indices_i32.To(core::Int64);

    core::Tensor active_keys = keys.IndexGet({active_indices}).To(host);

    std::unordered_map<std::string, core::Tensor> output;
    output.emplace("key", active_keys);
    output.emplace(
            "n_values",
            core::Tensor(
                    std::vector<int64_t>{static_cast<int64_t>(values.size())},
                    {1}, core::Int64, host));
    for (size_t i = 0; i < values.size(); ++i) {
        core::Tensor active_value_i =
                values[i].IndexGet({active_indices}).To(host);
        output.emplace(fmt::format("value_{:03d}", i), active_value_i);
    }

    std::string ext =
            utility::filesystem::GetFileExtensionInLowerCase(file_name);
    std::string postfix = ext != "npz" ? ".npz" : "";
    WriteNpz(file_name + postfix, output);
}

core::HashMap ReadHashMap(const std::string& file_name) {
    std::unordered_map<std::string, core::Tensor> tensor_map =
            t::io::ReadNpz(file_name);

    // Key
    core::Tensor keys = tensor_map.at("key");

    core::Dtype key_dtype = keys.GetDtype();

    core::SizeVector shape_key = keys.GetShape();
    core::SizeVector key_element_shape(shape_key.begin() + 1, shape_key.end());

    int64_t init_capacity = keys.GetLength();

    // Value(s)
    int64_t n_values = tensor_map.at("n_values")[0].Item<int64_t>();

    std::vector<core::Tensor> arr_input_values;
    std::vector<core::Dtype> dtypes_value;
    std::vector<core::SizeVector> element_shapes_value;

    for (int64_t i = 0; i < n_values; ++i) {
        core::Tensor value_i = tensor_map.at(fmt::format("value_{:03d}", i));

        core::Dtype value_dtype_i = value_i.GetDtype();

        core::SizeVector shape_value_i = value_i.GetShape();
        core::SizeVector value_element_shape_i(shape_value_i.begin() + 1,
                                               shape_value_i.end());

        arr_input_values.push_back(value_i);
        dtypes_value.push_back(value_dtype_i);
        element_shapes_value.push_back(value_element_shape_i);
    }

    auto hashmap =
            core::HashMap(init_capacity, key_dtype, key_element_shape,
                          dtypes_value, element_shapes_value, core::Device());

    core::Tensor masks, buf_indices;
    hashmap.Insert(keys, arr_input_values, masks, buf_indices);

    return hashmap;
}
}  // namespace io
}  // namespace t
}  // namespace open3d
