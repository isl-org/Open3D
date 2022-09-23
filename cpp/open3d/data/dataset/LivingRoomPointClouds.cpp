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
        Open3DDownloadsPrefix() + "redwood/livingroom1-fragments-ply.zip",
        "36e0eb23a66ccad6af52c05f8390d33e"};

LivingRoomPointClouds::LivingRoomPointClouds(const std::string& data_root)
    : DownloadDataset("LivingRoomPointClouds", data_descriptor, data_root) {
    paths_.reserve(57);
    for (int i = 0; i < 57; ++i) {
        paths_.push_back(GetExtractDir() + "/cloud_bin_" + std::to_string(i) +
                         ".ply");
    }
}

std::string LivingRoomPointClouds::GetPaths(size_t index) const {
    if (index > 56) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 56 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
