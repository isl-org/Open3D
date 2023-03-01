// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/pipelines/registration/Feature.h"

namespace open3d {
namespace io {

/// The general entrance for reading a Feature from a file
/// \return If the read function is successful.
bool ReadFeature(const std::string &filename,
                 pipelines::registration::Feature &feature);

/// The general entrance for writing a Feature to a file
/// \return If the write function is successful.
bool WriteFeature(const std::string &filename,
                  const pipelines::registration::Feature &feature);

bool ReadFeatureFromBIN(const std::string &filename,
                        pipelines::registration::Feature &feature);

bool WriteFeatureToBIN(const std::string &filename,
                       const pipelines::registration::Feature &feature);

}  // namespace io
}  // namespace open3d
