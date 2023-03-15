// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace open3d {
namespace utility {

/// Estimate the maximum number of threads to be used in a parallel region.
int EstimateMaxThreads();

/// Returns true if in an parallel section.
bool InParallel();

}  // namespace utility
}  // namespace open3d
