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

#include "open3d/t/io/HashmapIO.h"

#include "open3d/t/io/NumpyIO.h"
#include "open3d/utility/FileSystem.h"
namespace open3d {
namespace t {
namespace io {

void WriteHashmap(const std::string& file_name, const core::Hashmap& hashmap) {
    core::Tensor keys = hashmap.GetKeyTensor().To(core::HOST);
    core::Tensor values = hashmap.GetValueTensor().To(core::HOST);

    core::Tensor active_addrs;
    hashmap.GetActiveIndices(active_addrs);
    core::Tensor active_indices =
            active_addrs.To(core::HOST, core::Dtype::Int64);

    core::Tensor active_keys = keys.IndexGet({active_indices});
    core::Tensor active_values = values.IndexGet({active_indices});

    std::string ext =
            utility::filesystem::GetFileExtensionInLowerCase(file_name);
    std::string postfix = ext != "npz" ? ".npz" : "";
    WriteNpz(file_name + postfix,
             {{"key", active_keys}, {"value", active_values}});
}

core::Hashmap ReadHashmap(const std::string& file_name) {
    std::unordered_map<std::string, core::Tensor> tensor_map =
            t::io::ReadNpz(file_name);
    core::Tensor keys = tensor_map.at("key");
    core::Tensor values = tensor_map.at("value");

    core::Dtype dtype_key = keys.GetDtype();
    core::Dtype dtype_value = values.GetDtype();

    int length_key = keys.GetLength();
    int length_value = values.GetLength();
    if (length_key != length_value) {
        utility::LogError("Incompatible key length ({}) and value length ({}).",
                          length_key, length_value);
    }
    int init_capacity = length_key;

    core::SizeVector shape_key = keys.GetShape();
    core::SizeVector shape_value = values.GetShape();
    if (shape_key.size() < 2 || shape_value.size() < 2) {
        utility::LogError("key value element shape must be larger than 1 dim");
    }
    core::SizeVector element_shape_key(shape_key.begin() + 1, shape_key.end());
    core::SizeVector element_shape_value(shape_value.begin() + 1,
                                         shape_value.end());

    auto hashmap =
            core::Hashmap(init_capacity, dtype_key, dtype_value,
                          element_shape_key, element_shape_value, core::HOST);

    core::Tensor masks, addrs;
    hashmap.Insert(keys, values, masks, addrs);

    return hashmap;
}
}  // namespace io
}  // namespace t
}  // namespace open3d
