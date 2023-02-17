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

#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() + "20220301-data/MonkeyModel.zip",
        "fc330bf4fd8e022c1e5ded76139785d4"};

MonkeyModel::MonkeyModel(const std::string& data_root)
    : DownloadDataset("MonkeyModel", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    map_filename_to_path_ = {
            {"albedo", extract_dir + "/albedo.png"},
            {"ao", extract_dir + "/ao.png"},
            {"metallic", extract_dir + "/metallic.png"},
            {"monkey_material", extract_dir + "/monkey.mtl"},
            {"monkey_model", extract_dir + "/monkey.obj"},
            {"monkey_solid_material", extract_dir + "/monkey_solid.mtl"},
            {"monkey_solid_model", extract_dir + "/monkey_solid.obj"},
            {"normal", extract_dir + "/normal.png"},
            {"roughness", extract_dir + "/roughness.png"}};
}

}  // namespace data
}  // namespace open3d
