// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstddef>

namespace open3d {
namespace utility {

/// Estimate the maximum number of threads to be used in a parallel region.
int EstimateMaxThreads();

/// Returns a reference to the default grain size used by TBB.
/// Can be altered if needed.
std::size_t& DefaultGrainSizeTBB() noexcept;

/// Returns a reference to the default grain size used by TBB
/// for 2d blocked parallel ranges
/// Can be altered if needed
std::size_t& DefaultGrainSizeTBB2D() noexcept;

}  // namespace utility
}  // namespace open3d
