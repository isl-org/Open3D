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

#include "open3d/t/geometry/TensorMap.h"

#include <fmt/format.h>

#include <sstream>
#include <string>
#include <unordered_map>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {

bool TensorMap::IsSizeSynchronized() const {
    const int64_t primary_size = GetPrimarySize();
    for (auto& kv : *this) {
        if (kv.second.GetLength() != primary_size) {
            return false;
        }
    }
    return true;
}

void TensorMap::AssertPrimaryKeyInMapOrEmpty() const {
    if (size() != 0 && count(primary_key_) == 0) {
        utility::LogError("TensorMap does not contain primary key \"{}\".",
                          primary_key_);
    }
}

void TensorMap::AssertNoReservedKeys() const {
    const std::unordered_set<std::string>& reserved_keys = GetReservedKeys();
    for (const auto& kv : *this) {
        if (reserved_keys.count(kv.first)) {
            utility::LogError("TensorMap contains reserved key \"{}\".",
                              kv.first);
        }
    }
}

void TensorMap::AssertSizeSynchronized() const {
    if (!IsSizeSynchronized()) {
        const int64_t primary_size = GetPrimarySize();
        std::stringstream ss;
        ss << fmt::format("Primary Tensor \"{}\" has size {}, however: \n",
                          primary_key_, primary_size);
        for (auto& kv : *this) {
            if (kv.first != primary_key_ &&
                kv.second.GetLength() != primary_size) {
                fmt::format("    > Tensor \"{}\" has size {}.\n", kv.first,
                            kv.second.GetLength());
            }
        }
        utility::LogError("{}", ss.str());
    }
}

bool TensorMap::IsContiguous() const {
    for (const auto& kv : *this) {
        if (!kv.second.IsContiguous()) {
            return false;
        }
    }
    return true;
}

TensorMap TensorMap::Contiguous() const {
    TensorMap tensor_map_contiguous(GetPrimaryKey());
    for (const auto& kv : *this) {
        // If the tensor is contiguous, the underlying memory is used.
        tensor_map_contiguous[kv.first] = kv.second.Contiguous();
    }
    return tensor_map_contiguous;
}

std::unordered_set<std::string> TensorMap::GetReservedKeys() {
    const static std::unordered_set<std::string> reserved_keys = {
            // Python reserved key.
            "__class__",
            "__contains__",
            "__delattr__",
            "__delitem__",
            "__dir__",
            "__doc__",
            "__eq__",
            "__format__",
            "__ge__",
            "__getattribute__",
            "__getitem__",
            "__gt__",
            "__hash__",
            "__init__",
            "__init_subclass__",
            "__iter__",
            "__le__",
            "__len__",
            "__lt__",
            "__ne__",
            "__new__",
            "__reduce__",
            "__reduce_ex__",
            "__repr__",
            "__reversed__",
            "__setattr__",
            "__setitem__",
            "__sizeof__",
            "__str__",
            "__subclasshook__",
            "clear",
            "copy",
            "fromkeys",
            "get",
            "items",
            "keys",
            "pop",
            "popitem",
            "setdefault",
            "update",
            "values",
            // Custom reserved keys.
            "primary_key",
            "is_size_synchronized",
            "assert_size_synchronized",
    };
    return reserved_keys;
}

std::string TensorMap::ToString() const {
    const std::string primary_key = GetPrimaryKey();

    if (empty()) {
        return fmt::format("TensorMap(primary_key=\"{}\") with no attribute",
                           primary_key);
    }

    size_t max_key_len = 0;
    bool has_primary_key = false;
    std::vector<std::string> keys;
    keys.reserve(size());
    for (const auto& kv : *this) {
        const std::string key = kv.first;
        keys.push_back(key);
        max_key_len = std::max(max_key_len, key.size());
        if (key == primary_key) {
            has_primary_key = true;
        }
    }
    std::sort(keys.begin(), keys.end());

    const std::string tensor_format_str = fmt::format(
            "  - {{:<{}}}: shape={{}}, dtype={{}}, device={{}}", max_key_len);

    std::stringstream ss;
    ss << fmt::format("TensorMap(primary_key=\"{}\") with {} attribute{}:",
                      primary_key, size(), size() > 1 ? "s" : "")
       << std::endl;
    for (const std::string& key : keys) {
        const core::Tensor& val = at(key);
        ss << fmt::format(tensor_format_str, key, val.GetShape().ToString(),
                          val.GetDtype().ToString(),
                          val.GetDevice().ToString());
        if (key == primary_key) {
            ss << " (primary)";
        }
        ss << std::endl;
    }

    const std::string example_key = has_primary_key ? primary_key : keys[0];
    ss << fmt::format("  (Use . to access attributes, e.g., tensor_map.{})",
                      example_key);
    return ss.str();
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
