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

#include "open3d/io/ImageWarpingFieldIO.h"

#include <unordered_map>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"

namespace open3d {

namespace {
using namespace io;

bool ReadImageWarpingFieldFromJSON(
        const std::string &filename,
        pipelines::color_map::ImageWarpingField &warping_field) {
    return ReadIJsonConvertible(filename, warping_field);
}

bool WriteImageWarpingFieldToJSON(
        const std::string &filename,
        const pipelines::color_map::ImageWarpingField &warping_field) {
    return WriteIJsonConvertibleToJSON(filename, warping_field);
}

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           pipelines::color_map::ImageWarpingField &)>>
        file_extension_to_warping_field_read_function{
                {"json", ReadImageWarpingFieldFromJSON},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const pipelines::color_map::ImageWarpingField &)>>
        file_extension_to_warping_field_write_function{
                {"json", WriteImageWarpingFieldToJSON},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<pipelines::color_map::ImageWarpingField>
CreateImageWarpingFieldFromFile(const std::string &filename) {
    auto warping_field =
            std::make_shared<pipelines::color_map::ImageWarpingField>();
    ReadImageWarpingField(filename, *warping_field);
    return warping_field;
}

bool ReadImageWarpingField(
        const std::string &filename,
        pipelines::color_map::ImageWarpingField &warping_field) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_warping_field_read_function.find(filename_ext);
    if (map_itr == file_extension_to_warping_field_read_function.end()) {
        utility::LogWarning(
                "Read pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, warping_field);
}

bool WriteImageWarpingField(
        const std::string &filename,
        const pipelines::color_map::ImageWarpingField &trajectory) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_warping_field_write_function.find(filename_ext);
    if (map_itr == file_extension_to_warping_field_write_function.end()) {
        utility::LogWarning(
                "Write pipelines::color_map::ImageWarpingField failed: unknown "
                "file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, trajectory);
}

}  // namespace io
}  // namespace open3d
