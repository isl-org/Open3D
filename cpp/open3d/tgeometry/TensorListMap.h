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

#pragma once

#include <string>
#include <unordered_map>

#include "open3d/core/TensorList.h"

namespace open3d {
namespace tgeometry {

/// Map of string to TensorList. Provides helper function to maintain a
/// synchronized size (length) for the tensorlists.
///
/// The master key's tensorlist's size is used as the master size and master
/// device. Other tensorlist's size and device should be synchronized according
/// to the master.
class TensorListMap : public std::unordered_map<std::string, core::TensorList> {
public:
    /// Create empty TensorListMap and set master key.
    TensorListMap(const std::string& master_key)
        : std::unordered_map<std::string, core::TensorList>(),
          master_key_(master_key) {}

    /// Create TensorListMap with pre-populated values.
    TensorListMap(const std::string& master_key,
                  const std::unordered_map<std::string, core::TensorList>&
                          map_keys_to_tensorlists)
        : std::unordered_map<std::string, core::TensorList>(),
          master_key_(master_key) {
        Assign(map_keys_to_tensorlists);
    }

    /// A master key is always required.
    TensorListMap() = delete;

    /// Clear the current map and assign new keys and values. The master key
    /// remains unchanged. The input \p map_keys_to_tensorlists must at least
    /// contain the master key. Data won't be copied, tensorlists still share
    /// the same memory as the input.
    ///
    /// \param map_keys_to_tensorlists. The keys and values to be assigned.
    void Assign(const std::unordered_map<std::string, core::TensorList>&
                        map_keys_to_tensorlists);

    /// Synchronized push back, data will be copied. Before push back,
    /// IsSizeSynchronized() must be true.
    ///
    /// \param map_keys_to_tensors The keys and values to be pushed back. It
    /// must contain the same keys and each corresponding tensor must have the
    /// same dtype and device.
    void SynchronizedPushBack(
            const std::unordered_map<std::string, core::Tensor>&
                    map_keys_to_tensors);

    /// Returns the master key of the tensorlistmap.
    std::string GetMasterKey() const { return master_key_; }

    /// Returns true if all tensorlists in the map have the same size.
    bool IsSizeSynchronized() const;

    /// Assert IsSizeSynchronized().
    void AssertSizeSynchronized() const;

    /// Returns true if the key exists in the map.
    /// Same as C++20's std::unordered_map::contains().
    bool Contains(const std::string& key) const { return count(key) != 0; }

private:
    /// Asserts that \p map_keys_to_tensors has the same keys as the
    /// TensorListMap.
    ///
    /// \param map_keys_to_tensors A map of string to Tensor. Typically the map
    /// is used for SynchronizedPushBack.
    void AssertTensorMapSameKeys(
            const std::unordered_map<std::string, core::Tensor>&
                    map_keys_to_tensors) const;

    /// Asserts that all of the tensors in \p map_keys_to_tensors have the same
    /// device as the master tensorlist.
    ///
    /// \param map_keys_to_tensors A map of string to Tensor. Typically the map
    /// is used for SynchronizedPushBack.
    void AssertTensorMapSameDevice(
            const std::unordered_map<std::string, core::Tensor>&
                    map_keys_to_tensors) const;

    /// Returns the size (length) of the master key's tensorlist.
    int64_t GetMasterSize() const { return at(master_key_).GetSize(); }

    /// Returns the device of the master key's tensorlist.
    core::Device GetMasterDevice() const { return at(master_key_).GetDevice(); }

    /// The master key's tensorlist's size is used as the master size and master
    /// device. Other tensorlist's size and device should be synchronized
    /// according to the master.
    std::string master_key_ = "UNDEFINED";
};

}  // namespace tgeometry
}  // namespace open3d
