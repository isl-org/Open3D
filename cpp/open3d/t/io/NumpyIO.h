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
