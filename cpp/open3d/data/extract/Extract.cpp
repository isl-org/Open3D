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

#include "open3d/data/extract/Extract.h"

#include <unordered_map>

#include "open3d/data/extract/Zip.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/ProgressReporters.h"

namespace open3d {
namespace data {

static const std::unordered_map<std::string,
                                std::function<bool(const std::string&,
                                                   const std::string&,
                                                   const std::string&,
                                                   const bool always_overwrite,
                                                   const bool print_progress)>>
        file_extension_to_extract_function{
                {"zip", ExtractFromZIP},
        };

bool Extract(const std::string& filename,
             const std::string& extract_dir,
             const std::string& password,
             const bool always_overwrite,
             const bool print_progress) {
    std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(filename);

    utility::LogDebug("Format {} File {}", format, filename);

    auto map_itr = file_extension_to_extract_function.find(format);
    if (map_itr == file_extension_to_extract_function.end()) {
        utility::LogWarning(
                "Extraction failed: unknown file extension for "
                "{} (format: {}).",
                filename, format);
        return false;
    }

    bool success = map_itr->second(filename, extract_dir, password,
                                   always_overwrite, print_progress);

    utility::LogDebug("Extraction Successful.");
    return success;
}

}  // namespace data
}  // namespace open3d
