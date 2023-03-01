// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/core/Tensor.h"
#include "open3d/core/hashmap/HashMap.h"

namespace open3d {
namespace t {
namespace io {

/// Read a hash map's keys and values from a npz file at 'key' and 'value'.
/// Return a hash map on CPU.
///
/// \param filename The npz file name to read from.
core::HashMap ReadHashMap(const std::string& filename);

/// Save a hash map's keys and values to a npz file at 'key' and 'value'.
///
/// \param filename The npz file name to write to.
/// \param hashmap HashMap to save.
void WriteHashMap(const std::string& filename, const core::HashMap& hashmap);

}  // namespace io
}  // namespace t
}  // namespace open3d
