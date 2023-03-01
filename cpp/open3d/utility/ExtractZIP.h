// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Do not include this in public facing header.

#pragma once

#include <string>

namespace open3d {
namespace utility {

/// \brief Function to extract files compressed in `.zip` format.
/// \param file_path Path to file. Example: "/path/to/file/file.zip"
/// \param extract_dir Directory path where the file will be extracted to.
void ExtractFromZIP(const std::string& file_path,
                    const std::string& extract_dir);

}  // namespace utility
}  // namespace open3d
