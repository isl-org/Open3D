// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace open3d {
namespace utility {

/// \brief Returns true if the file is a supported compressed file path. It does
/// not check if the file exists. It only checks the file extension.
/// \param file_path The file path to check.
bool IsSupportedCompressedFilePath(const std::string& file_path);

/// \brief Function to extract compressed files.
/// \param file_path Path to file. Example: "/path/to/file/file.zip"
/// \param extract_dir Directory path where the file will be extracted to. If
/// the directory does not exist, it will be created.
void Extract(const std::string& file_path, const std::string& extract_dir);

}  // namespace utility
}  // namespace open3d
