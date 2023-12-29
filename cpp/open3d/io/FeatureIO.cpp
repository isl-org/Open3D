// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/FeatureIO.h"

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace io {

bool ReadFeature(const std::string &filename,
                 pipelines::registration::Feature &feature) {
    return ReadFeatureFromBIN(filename, feature);
}

bool WriteFeature(const std::string &filename,
                  const pipelines::registration::Feature &feature) {
    return WriteFeatureToBIN(filename, feature);
}

}  // namespace io
}  // namespace open3d
