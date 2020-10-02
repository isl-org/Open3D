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

#include "open3d/t/geometry/TensorListMap.h"

#include <fmt/format.h>

#include <sstream>
#include <string>
#include <unordered_map>

#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {

void TensorListMap::Assign(
        const std::unordered_map<std::string, core::TensorList>&
                map_keys_to_tensorlists) {
    if (!map_keys_to_tensorlists.count(primary_key_)) {
        utility::LogError(
                "The input tensorlist map does not contain the primary key {}.",
                primary_key_);
    }
    this->clear();
    const core::Device& primary_device =
            map_keys_to_tensorlists.at(primary_key_).GetDevice();
    for (auto& kv : map_keys_to_tensorlists) {
        if (primary_device != kv.second.GetDevice()) {
            utility::LogError(
                    "Primary tensorlist has device {}, however, another "
                    "tensorlist has device {}.",
                    primary_device.ToString(),
                    kv.second.GetDevice().ToString());
        }
        this->operator[](kv.first) = kv.second;
    }
}

bool TensorListMap::IsSizeSynchronized() const {
    for (auto& kv : *this) {
        if (kv.second.GetSize() != GetPrimarySize()) {
            return false;
        }
    }
    return true;
}

void TensorListMap::AssertSizeSynchronized() const {
    if (!IsSizeSynchronized()) {
        std::stringstream ss;
        ss << fmt::format("Primary TensorList \"{}\" has size {}, however: \n",
                          primary_key_, GetPrimarySize());
        for (auto& kv : *this) {
            if (kv.first != primary_key_ &&
                kv.second.GetSize() != GetPrimarySize()) {
                fmt::format("    > TensorList \"{}\" has size {}.\n", kv.first,
                            kv.second.GetSize());
            }
        }
        utility::LogError("{}", ss.str());
    }
}

void TensorListMap::SynchronizedPushBack(
        const std::unordered_map<std::string, core::Tensor>&
                map_keys_to_tensors) {
    AssertSizeSynchronized();
    AssertTensorMapSameKeys(map_keys_to_tensors);
    AssertTensorMapSameDevice(map_keys_to_tensors);
    for (auto& kv : map_keys_to_tensors) {
        at(kv.first).PushBack(kv.second);
    }
}

void TensorListMap::AssertTensorMapSameKeys(
        const std::unordered_map<std::string, core::Tensor>&
                map_keys_to_tensors) const {
    bool is_same = true;
    if (size() != map_keys_to_tensors.size()) {
        is_same = false;
    } else {
        for (auto& kv : map_keys_to_tensors) {
            if (!Contains(kv.first)) {
                is_same = false;
                break;
            }
        }
    }
    if (!is_same) {
        utility::LogError(
                "The input map does not have the same keys as the primary "
                "tensorlist.");
    }
}

void TensorListMap::AssertTensorMapSameDevice(
        const std::unordered_map<std::string, core::Tensor>&
                map_keys_to_tensors) const {
    const core::Device& primary_device = GetPrimaryDevice();
    for (auto& kv : map_keys_to_tensors) {
        if (kv.second.GetDevice() != primary_device) {
            utility::LogError(
                    "Tensor in the input map does not have the same device as "
                    "the primary tensorlist: {} != {}.",
                    kv.second.GetDevice().ToString(),
                    primary_device.ToString());
        }
    }
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
