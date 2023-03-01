// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <unordered_map>

#include "open3d/core/Tensor.h"

namespace open3d {
namespace t {
namespace io {

/// Read Numpy .npy file to a tensor.
///
/// \param file_name The file name to read from.
core::Tensor ReadNpy(const std::string& file_name);

/// Save a tensor to a Numpy .npy file.
///
/// \param file_name The file name to write to.
/// \param tensor The tensor to save.
void WriteNpy(const std::string& file_name, const core::Tensor& tensor);

/// Read Numpy .npz file to an unordered_map from string to tensor.
///
/// \param file_name The file name to read from.
std::unordered_map<std::string, core::Tensor> ReadNpz(
        const std::string& file_name);

/// Save a string to tensor map as Numpy .npz file.
///
/// \param file_name The file name to write to.
/// \param tensor_map The tensor map to save.
void WriteNpz(const std::string& file_name,
              const std::unordered_map<std::string, core::Tensor>& tensor_map);

}  // namespace io
}  // namespace t
}  // namespace open3d
