// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>
#include <vector>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() + "20220301-data/SampleL515Bag.zip",
        "9770eeb194c78103037dbdbec78b9c8c"};

SampleL515Bag::SampleL515Bag(const std::string& data_root)
    : DownloadDataset("SampleL515Bag", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/L515_test_s.bag";
}

}  // namespace data
}  // namespace open3d
