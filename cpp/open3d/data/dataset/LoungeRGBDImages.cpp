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
        Open3DDownloadsPrefix() + "20220301-data/LoungeRGBDImages.zip",
        "cdd307caef898519a8829ce1b6ab9f75"};

LoungeRGBDImages::LoungeRGBDImages(const std::string& data_root)
    : DownloadDataset("LoungeRGBDImages", data_descriptor, data_root) {
    color_paths_.reserve(3000);
    depth_paths_.reserve(3000);
    const std::string extract_dir = GetExtractDir();
    const size_t n_zero = 6;
    for (int i = 1; i < 3000; ++i) {
        std::string idx = std::to_string(i);
        idx = std::string(n_zero - std::min(n_zero, idx.length()), '0') + idx;
        color_paths_.push_back(extract_dir + "/color/" + idx + ".png");
        depth_paths_.push_back(extract_dir + "/depth/" + idx + ".png");
    }

    trajectory_log_path_ = extract_dir + "/lounge_trajectory.log";
    reconstruction_path_ = extract_dir + "/lounge.ply";
}

}  // namespace data
}  // namespace open3d
