// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <string>

#include "open3d/data/Dataset.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace data {

const static DataDescriptor data_descriptor = {
        Open3DDownloadsPrefix() + "20220301-data/JackJackL515Bag.bag",
        "9f670dc92569b986b739c4179a659176"};

JackJackL515Bag::JackJackL515Bag(const std::string& data_root)
    : DownloadDataset("JackJackL515Bag", data_descriptor, data_root) {
    path_ = GetExtractDir() + "/JackJackL515Bag.bag";
}

}  // namespace data
}  // namespace open3d
