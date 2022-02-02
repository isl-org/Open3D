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

void Extract(const std::string& file_path, const std::string& extract_dir) {
    const std::string format =
            utility::filesystem::GetFileExtensionInLowerCase(file_path);
    utility::LogInfo("Extracting {}.", file_path);

    if (file_extension_to_extract_function.count(format) == 0) {
        utility::LogError(
                "Extraction Failed: unknown file extension for "
                "{} (format: {}).",
                file_path, format);
    }

    file_extension_to_extract_function.at(format)(file_path, extract_dir);
    utility::LogInfo("Extracted to {}.", extract_dir);
}

}  // namespace utility
}  // namespace open3d
