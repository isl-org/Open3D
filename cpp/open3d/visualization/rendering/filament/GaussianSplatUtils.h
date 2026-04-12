// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Shared integer math utilities used by the Gaussian splatting compute backend.

#pragma once

#include <algorithm>
#include <cstdint>

namespace open3d {
namespace visualization {
namespace rendering {

/// Integer ceiling division: ceil(value / divisor).
inline int CeilDiv(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

/// Unsigned integer ceiling division: max(1, ceil(n / denom)).
inline std::uint32_t DivUp(std::uint32_t n, std::uint32_t denom) {
    return std::max(1u, (n + denom - 1u) / denom);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
