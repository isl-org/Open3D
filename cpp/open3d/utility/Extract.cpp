// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Extract.h"

#include <unordered_map>

#include "open3d/utility/ExtractZIP.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

static const std::unordered_map<
        std::string,
        std::function<void(const std::string&, const std::string&)>>
        file_extension_to_extract_function{
                {"zip", ExtractFromZIP},
        };

bool IsSupportedCompressedFilePath(const std::string& file_path) {
    const std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(file_path);
    return file_extension_to_extract_function.count(format) != 0;
}

void Extract(const std::string& file_path, const std::string& extract_dir) {
    if (!utility::filesystem::FileExists(file_path)) {
        utility::LogError("File {} does not exist.", file_path);
    }
    if (!IsSupportedCompressedFilePath(file_path)) {
        utility::LogError("Extraction Failed: unknown extension for {}.",
                          file_path);
    }
    if (!utility::filesystem::DirectoryExists(extract_dir)) {
        utility::filesystem::MakeDirectoryHierarchy(extract_dir);
        utility::LogInfo("Created directory {}.", extract_dir);
    }
    utility::LogInfo("Extracting {}.", file_path);
    const std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(file_path);
    file_extension_to_extract_function.at(format)(file_path, extract_dir);
    utility::LogInfo("Extracted to {}.", extract_dir);
}

}  // namespace utility
}  // namespace open3d
