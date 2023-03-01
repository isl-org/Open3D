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

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() +
                "doppler-icp-data/carla-town05-curved-walls.zip",
        "73a9828fb7790481168124c02398ee01"};

DemoDopplerICPSequence::DemoDopplerICPSequence(const std::string& data_root)
    : DownloadDataset("DemoDopplerICPSequence", data_descriptor, data_root) {
    for (int i = 1; i <= 10; ++i) {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << i;
        paths_.push_back(GetExtractDir() + "/xyzd_sequence/" + ss.str() +
                         ".xyzd");
    }

    calibration_path_ = GetExtractDir() + "/calibration.json";
    trajectory_path_ = GetExtractDir() + "/ground_truth_poses.txt";
}

std::string DemoDopplerICPSequence::GetPath(std::size_t index) const {
    if (index > 9) {
        utility::LogError(
                "Invalid index. Expected index between 0 to 9 but got {}.",
                index);
    }
    return paths_[index];
}

}  // namespace data
}  // namespace open3d
