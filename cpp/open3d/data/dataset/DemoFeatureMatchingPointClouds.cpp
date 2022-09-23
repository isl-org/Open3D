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
        Open3DDownloadsPrefix() +
                "20220201-data/DemoFeatureMatchingPointClouds.zip",
        "02f0703ce0cbf4df78ce2602ae33fc79"};

DemoFeatureMatchingPointClouds::DemoFeatureMatchingPointClouds(
        const std::string& data_root)
    : DownloadDataset(
              "DemoFeatureMatchingPointClouds", data_descriptor, data_root) {
    const std::string extract_dir = GetExtractDir();
    point_cloud_paths_ = {extract_dir + "/cloud_bin_0.pcd",
                          extract_dir + "/cloud_bin_1.pcd"};
    fpfh_feature_paths_ = {extract_dir + "/cloud_bin_0.fpfh.bin",
                           extract_dir + "/cloud_bin_1.fpfh.bin"};
    l32d_feature_paths_ = {extract_dir + "/cloud_bin_0.d32.bin",
                           extract_dir + "/cloud_bin_1.d32.bin"};
}

}  // namespace data
}  // namespace open3d
