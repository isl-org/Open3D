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

#include "open3d/t/geometry/TensorMap.h"

#include <sstream>
#include <string>
#include <unordered_map>

#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {

void TensorMap::Assign(const std::unordered_map<std::string, core::Tensor>&
                               map_keys_to_tensors) {
    if (!map_keys_to_tensors.count(primary_key_)) {
        utility::LogError("primary key {} must be present in the input map.",
                          primary_key_);
    }
    this->clear();
    const core::Device& primary_device =
            map_keys_to_tensors.at(primary_key_).GetDevice();
    for (auto& kv : map_keys_to_tensors) {
        if (primary_device != kv.second.GetDevice()) {
            utility::LogError(
                    "Primary tensor has device {}, however, another tensor has "
                    "device {}.",
                    primary_device.ToString(),
                    kv.second.GetDevice().ToString());
        }
        this->operator[](kv.first) = kv.second;
    }
}

bool TensorMap::IsSizeSynchronized() const {
    const int64_t primary_size = GetPrimarySize();
    for (auto& kv : *this) {
        if (kv.second.GetShape()[0] != primary_size) {
            return false;
        }
    }
    return true;
}

void TensorMap::AssertSizeSynchronized() const {
    if (!IsSizeSynchronized()) {
        const int64_t primary_size = GetPrimarySize();
        std::stringstream ss;
        ss << fmt::format("Primary Tensor \"{}\" has size {}, however: \n",
                          primary_key_, primary_size);
        for (auto& kv : *this) {
            if (kv.first != primary_key_ &&
                kv.second.GetShape()[0] != primary_size) {
                fmt::format("    > Tensor \"{}\" has size {}.\n", kv.first,
                            kv.second.GetShape()[0]);
            }
        }
        utility::LogError("{}", ss.str());
    }
}

void TensorMap::AssertTensorMapSameDevice(
        const std::unordered_map<std::string, core::Tensor>&
                map_keys_to_tensors) const {
    const core::Device& primary_device = GetPrimaryDevice();
    for (auto& kv : map_keys_to_tensors) {
        if (kv.second.GetDevice() != primary_device) {
            utility::LogError(
                    "Tensor in the input map does not have the same device as "
                    "the primary Tensor: {} != {}.",
                    kv.second.GetDevice().ToString(),
                    primary_device.ToString());
        }
    }
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
